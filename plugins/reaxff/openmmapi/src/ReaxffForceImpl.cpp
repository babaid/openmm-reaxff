#include "openmm/internal/ReaxffForceImpl.h"
#include "openmm/internal/PuremdInterface.h"

#include "openmm/OpenMMException.h"
#include "openmm/Units.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/kernels.h"

#include "omp.h"

#include <algorithm>
#include <limits>

using namespace OpenMM;
using namespace std;

constexpr size_t parallel_threshold = 100;

// flattens the QM atom positions, converts them to angstroms and sets their
// charges  in the context to 0
inline void transformPosQM(const std::vector<Vec3> &positions,
                           const std::vector<int>   indices,
                           std::vector<double>     &out)
{
    out.resize(indices.size() * 3);

#pragma omp parallel for if (indices.size() > parallel_threshold)

    for (size_t i = 0; i < indices.size(); ++i)
    {
        int Index      = indices[i];
        out[i * 3]     = positions[indices[i]][0] * AngstromsPerNm;
        out[i * 3 + 1] = positions[indices[i]][1] * AngstromsPerNm;
        out[i * 3 + 2] = positions[indices[i]][2] * AngstromsPerNm;
    }
}

// flattens the MM atom positions converts them to angstroms and converts their
// charges to coulombs
inline void transformPosqMM(const std::vector<Vec3>   &positions,
                            const std::vector<double> &charges,
                            const std::vector<int>    &indices,
                            std::vector<double>       &out)
{
    out.resize(indices.size() * 4);

#pragma omp parallel for if (indices.size() > parallel_threshold)
    for (size_t i = 0; i < indices.size(); ++i)
    {
        out[i * 4]     = positions[indices[i]][0] * AngstromsPerNm;
        out[i * 4 + 1] = positions[indices[i]][1] * AngstromsPerNm;
        out[i * 4 + 2] = positions[indices[i]][2] * AngstromsPerNm;
        out[i * 4 + 3] = charges[indices[i]];
    }
}

// filters the all atom symbols list based on indices we want
inline void getSymbolsByIndex(const std::vector<char> &symbols,
                              const std::vector<int>  &indices,
                              std::vector<char>       &out)
{
    out.resize(indices.size() * 2);

#pragma omp parallel for if (indices.size() > parallel_threshold)
    for (size_t i = 0; i < indices.size(); ++i)
    {
        out[i * 2]     = symbols[indices[i] * 2];
        out[i * 2 + 1] = symbols[indices[i] * 2 + 1];
    }
}

// gets the lenghts of the periodic box sides and the angles.
inline void getBoxInfo(const std::vector<Vec3> &positions,
                       std::vector<double>     &simBoxInfo)
{
    // the box will be of the size of the molecule plus a 2nm cutoff.
    double min, max; 
    double cutoff = 2.0;

    for (int i = 0; i < 3; i++)
    {
        min = std::numeric_limits<double>::infinity();
        max = -std::numeric_limits<double>::infinity();
        for (const auto &pos : positions)
        {
            max = std::max(max, pos[i]);
            min = std::min(min, pos[i]);
        }
        simBoxInfo[i] = (max - min + cutoff) * AngstromsPerNm;
    }
    simBoxInfo[3] = simBoxInfo[4] = simBoxInfo[5] = 90.0;
}

inline std::pair<Vec3, Vec3>
calculateBoundingBox(const std::vector<Vec3> &positions,
                     const std::vector<int> &Indices, double bbCutoff)
{
    Vec3 cutoff    = {bbCutoff, bbCutoff, bbCutoff};
    Vec3 minBounds = {std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max()};
    Vec3 maxBounds = {std::numeric_limits<double>::lowest(),
                      std::numeric_limits<double>::lowest(),
                      std::numeric_limits<double>::lowest()};

    for (const auto &Index : Indices)
    {
        for (int i = 0; i < 3; ++i)
        {
            minBounds[i] = std::min(minBounds[i], positions[Index][i]);
            maxBounds[i] = std::max(maxBounds[i], positions[Index][i]);
        }
    }

    return {minBounds - cutoff, maxBounds + cutoff};
}

inline bool isPointInsideBoundingBox(const Vec3                  &point,
                                     const std::pair<Vec3, Vec3> &boundingBox)
{
    // returns false if a point is outside of the bounding box
    const auto &[minBounds, maxBounds] = boundingBox;
    for (int i = 0; i < 3; i++)
    {
        if ((point[i] < minBounds[i]) || (point[i] > maxBounds[i]))
        {
            return false;
        }
    }
    return true;
}

// this function is supposed to filter the relevant MM atoms,
// which should considerably speed up the calculations on the PuReMD side
//  This function is O(n)
inline void filterMMAtoms(const std::vector<Vec3>     &positions,
                          const std::vector<int>      &mmIndices,
                          const std::pair<Vec3, Vec3> &bbCog,
                          std::vector<int>            &relevantIndices)
{
    const int                     numThreads = omp_get_num_threads();
    std::vector<std::vector<int>> localIndices(numThreads);

    // Parallel section to filter indices
#pragma omp parallel num_threads(numThreads)
    {
        int               threadId = omp_get_thread_num();
        std::vector<int> &localVec = localIndices[threadId];

        // Process each index in mmIndices
#pragma omp for
        for (size_t i = 0; i < mmIndices.size(); i++)
        {
            const Vec3 &point = positions[mmIndices[i]];
            if (isPointInsideBoundingBox(point, bbCog))
            {
                localVec.push_back(mmIndices[i]);
            }
        }
    }

    // Merge the results from all threads into relevantIndices
    for (const auto &localVec : localIndices)
    {
        // Add each local vector's elements to relevantIndices
        relevantIndices.insert(relevantIndices.end(), localVec.begin(),
                               localVec.end());
    }
}

ReaxffForceImpl::ReaxffForceImpl(const ReaxffForce &owner)
    : CustomCPPForceImpl(owner), owner(owner)
{
    std::string ffield_file, control_file;
    owner.getFileNames(ffield_file, control_file);
    Interface.setInputFileNames(ffield_file, control_file);
    owner.getNeighborListUpdateInterval(neighborlistUpdateInterval);

    for (int i = 0; i < owner.getNumAtoms(); ++i)
    {
        int    particle;
        char   symbol[2];
        int    isqm;
        double charge;
        owner.getParticleParameters(i, particle, symbol, charge, isqm);
        if (isqm)
        {
            qmParticles.emplace_back(particle);
            qmSymbols.emplace_back(symbol[0]);
            qmSymbols.emplace_back(symbol[1]);
        }
        else
        {
            mmParticles.emplace_back(particle);
        }
        AtomSymbols.emplace_back(symbol[0]);
        AtomSymbols.emplace_back(symbol[1]);
        charges.emplace_back(charge);
    }
    for (int i = 0; i<owner.getNumLinkAtoms(); i++)
    {
        int particle1, particle2;
        owner.getLinkAtoms(i, particle1, particle2);
        linkAtoms.push_back({particle1, particle2});
        linkAtomPositions.push_back(Vec3(0.0, 0.0, 0.0));
    }
}
double ReaxffForceImpl::computeForce(ContextImpl             &context,
                                     const std::vector<Vec3> &positions,
                                     std::vector<Vec3>       &forces)
{
    int N     = owner.getNumAtoms();
    int numQm = qmParticles.size();

    std::vector<double> qmPos, mmPos_q;
    std::vector<double> simBoxInfo(6);
    // set up link atoms...
    

    for (int i=0; i< linkAtoms.size(); i++)
    {
        auto link = linkAtoms[i];
        Vec3 RL = positions[link.first] + 0.723*(positions[link.second]-positions[link.first]);
        linkAtomPositions[i] = RL;
    }

    // box info for reaxff
    getBoxInfo(positions, simBoxInfo);

    // split QM atom positions and MM atom positions + charges
    transformPosQM(positions, qmParticles, qmPos);

    for (int i=0; i< linkAtoms.size(); i++)
    {
        qmPos.push_back(linkAtomPositions[i*3]);
        qmPos.push_back(linkAtomPositions[i*3 + 1]);
        qmPos.push_back(linkAtomPositions[i*3 + 2]);
        qmSymbols.push_back("H");
        qmSymbols.push_back("\0");
    }

    // get relevant MM indices from a bounding box sorrounding the ReaxFF atoms
    //  ~1nm makes sense as it is the upper taper radius in the ReaxFF
    //  nonbondonded potentials
    double                bbCutoff = 1.0;
    std::pair<Vec3, Vec3> bbCog =
        calculateBoundingBox(positions, qmParticles, bbCutoff);

    //std::vector<int> relevantMMIndices;
    if (callCounter%neighborlistUpdateInterval==0)
    {
        relevantMMIndices.clear();
        filterMMAtoms(positions, mmParticles, bbCog, relevantMMIndices);
    }

    std::vector<char> mmAtomSymbols;
    int               numMMAtoms = relevantMMIndices.size();

    getSymbolsByIndex(AtomSymbols, relevantMMIndices, mmAtomSymbols);
    transformPosqMM(positions, charges, relevantMMIndices, mmPos_q);
    // OUTPUT
    std::vector<double> qmForces(numQm * 3, 0), mmForces(numMMAtoms * 3, 0);
    std::vector<double> qmQ(numQm, 0);
    double              energy;

    Interface.getReaxffPuremdForces(numQm, qmSymbols, qmPos, numMMAtoms,
                                    mmAtomSymbols, mmPos_q, simBoxInfo,
                                    qmForces, mmForces, qmQ, energy);
    // Merge QM and MM forces
    std::vector<Vec3> transformedForces(owner.getNumAtoms(), {0.0, 0.0, 0.0});

#pragma omp parallel for if (qmParticles.size() > parallel_threshold)
    for (size_t i = 0; i < qmParticles.size(); ++i)
    {
        transformedForces[qmParticles[i]][0] = qmForces[3 * i];
        transformedForces[qmParticles[i]][1] = qmForces[3 * i + 1];
        transformedForces[qmParticles[i]][2] = qmForces[3 * i + 2];
    }
// distribute forces of link atoms between qm and mm atom using lever rule.
for (size_t i = 0 i < linkAtoms.size(); i++)
{
    auto qmAtom = linkAtoms[i].first;
    auto mmAtom = linkAtoms[i].second;
    Vec3 rqlv = linkAtomPositions[i] - positions[qmAtom];
    Vec3 rmqv = positions[qmAtom] - positions[mmAtom];
    double rql = std::sqrt(rqlv.dot(rqlv));
    double rmq = std::sqrt(rmqv.dot(rmqv));
    double scal = rql/rmq;
    double nscal = 1-scal;
    transformedForces[qmAtom][0] += nscal*qmForces[(qmParticles.size() + i) * 3];
    transformedForces[qmAtom][1] += nscal*qmForces[(qmParticles.size() + i) * 3 + 1];
    transformedForces[qmAtom][2] += nscal*qmForces[(qmParticles.size() + i) * 3 + 2];
    transformedForces[mmAtom][0] += scal * qmForces[(qmParticles.size() + i) * 3];
    transformedForces[mmAtom][1] += scal * qmForces[(qmParticles.size() + i) * 3 + 1];
    transformedForces[mmAtom][2] += scal * qmForces[(qmParticles.size() + i) * 3 + 2];
}

#pragma omp parallel for if (relevantMMIndices.size() > parallel_threshold)
    for (size_t i = 0; i < relevantMMIndices.size(); ++i)
    {
        transformedForces[relevantMMIndices[i]][0] = mmForces[i * 3];
        transformedForces[relevantMMIndices[i]][1] = mmForces[i * 3 + 1];
        transformedForces[relevantMMIndices[i]][2] = mmForces[i * 3 + 2];
    }

#pragma omp parallel for if (AtomSymbols.size() > parallel_threshold)
    for (size_t i = 0; i < forces.size(); ++i)
    {
        forces[i] = -transformedForces[i]*AngstromsPerNm*KJPerKcal*0.1;
    }
    callCounter++;
    return energy*KJPerKcal*0.1;
}
