//
// Created by babaid on 05.10.24.
//

#include "openmm/internal/ReaxffForceImpl.h"
#include "openmm/internal/PuremdInterface.h"

#include "openmm/OpenMMException.h"
#include "openmm/Units.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/kernels.h"
#include "omp.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>

using namespace OpenMM;
using namespace std;

constexpr double EvPerNmToKcalPerNm = 32.2063782682;
constexpr double ProtonToCoulomb  = 1.602E-19;

constexpr size_t parallel_threshold = 100;

// flattens the QM atom positions, converts them to angstroms and sets their
// charges  in the context to 0
inline void transformPosQM(const std::vector<Vec3> &positions,
                           const std::vector<int> indices,
                           std::vector<double> &out)
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
inline void transformPosqMM(const std::vector<Vec3> &positions,
                            const std::vector<double> &charges,
                            const std::vector<int> &indices,
                            std::vector<double> &out)
{
    out.resize(indices.size() * 4);

    #pragma omp parallel for
    for (size_t i = 0; i < indices.size(); ++i)
    {
        out[i * 4]     = positions[indices[i]][0] * AngstromsPerNm;
        out[i * 4 + 1] = positions[indices[i]][1] * AngstromsPerNm;
        out[i * 4 + 2] = positions[indices[i]][2] * AngstromsPerNm;
        out[i * 4 + 3] = charges[indices[i]] * ProtonToCoulomb;
    }
}

// filters the all atom symbols list based on indices we want
inline void getSymbolsByIndex(const std::vector<char> &symbols,
                              const std::vector<int> &indices,
                              std::vector<char> &out)
{
    out.resize(indices.size() * 2);
    
    #pragma omp parallel for
    for (size_t i = 0; i < indices.size(); ++i)
    {
        out[i * 2]     = symbols[indices[i] * 2];
        out[i * 2 + 1] = symbols[indices[i] * 2 + 1];
    }
}

// gets the lenghts of the periodic box sides and the angles.
inline void getBoxInfo(const std::vector<Vec3> &positions, std::vector<double> &simBoxInfo)
{
    // the box will be of the size of the molecule plus a 2nm cutoff.
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    
    for (int i = 0; i < 3; i++)
    {
        for (const auto &pos : positions)
        {
            max = std::max(max, pos[i]);
            min = std::min(min, pos[i]);
        }
        simBoxInfo[i] = max * AngstromsPerNm - min * AngstromsPerNm + 2 * AngstromsPerNm;
    }
    simBoxInfo[3] = simBoxInfo[4] = simBoxInfo[5] = 90.0;
}

inline std::pair<Vec3, Vec3> calculateBoundingBox(const std::vector<Vec3> &positions,
                                                  const std::vector<int> &Indices,
                                                  double bbCutoff)
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

bool isPointInsideBoundingBox(const Vec3 &point, const std::pair<Vec3, Vec3> &boundingBox)
{
    // returns false if a point is outside of the bounding box
    const auto &[minBounds, maxBounds] = boundingBox;
    for (int i=0;i<3; i++)
    {
        if ( (point[i] < minBounds[i]) || (point[i] > maxBounds[i]))
        {
            return false;
        }
    }
    return true;
}

// this function is supposed to filter the relevant MM atoms,
// which should considerably speed up the calculations on the PuReMD side
//  This function is O(n)
inline void filterMMAtomsOMP(const std::vector<Vec3> &positions,
                             const std::vector<int> &mmIndices,
                             const std::pair<Vec3, Vec3> &bbCog,
                             std::vector<int> &relevantIndices)
{
    const int numThreads = omp_get_num_threads();
    std::vector<std::vector<int>> localIndices(numThreads);

    // Parallel section to filter indices
#pragma omp parallel num_threads(numThreads)
    {
        int threadId = omp_get_thread_num();
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
        relevantIndices.insert(relevantIndices.end(), localVec.begin(), localVec.end());
    }
}


ReaxffForceImpl::ReaxffForceImpl(const ReaxffForce &owner)
    : CustomCPPForceImpl(owner), owner(owner)
{
    std::string ffield_file, control_file;
    owner.getFileNames(ffield_file, control_file);
    Interface.setInputFileNames(ffield_file, control_file);

    for (int i = 0; i < owner.getNumAtoms(); ++i)
    {
        int particle;
        char symbol[2];
        int isqm;
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
        // instead of MM symbols, because we have to filter this regularly
        AtomSymbols.emplace_back(symbol[0]);
        AtomSymbols.emplace_back(symbol[1]);
        charges.emplace_back(charge);
    }
}

double ReaxffForceImpl::computeForce(ContextImpl &context,
                                             const std::vector<Vec3> &positions,
                                             std::vector<Vec3> &forces)
{
    // double factor = context.getReaxffTemperatureRatio();

    // need to seperate positions
    // next we need to seperate and flatten the QM/MM positions and convert to AA#
    int N     = owner.getNumAtoms();
    int numQm = qmParticles.size();
    std::vector<double> qmPos, mmPos_q;
    // get the box size. move this into a function
    std::vector<double> simBoxInfo(6);
    getBoxInfo(positions, simBoxInfo);

    // retrieve charges from the context. Had to introduce some changes to classes
    // Context, ContextImpl,
    //  UpdateStateDataKernel, CommonUpdateStateDataKernel
    
    // flatten relevant qm positions and set charges to 0. The last step is
    // important so no exclusions have to be set manually.
    transformPosQM(positions, qmParticles, qmPos);

    // get relevant MM indices from a bounding box sorrounding the ReaxFF atoms
    //  ~1nm makes total sense as it is the upper taper radius, so interactions
    //  will be 0 anyways further away
    double bbCutoff = 1.0;
    std::vector<int> relevantMMIndices;

    std::pair<Vec3, Vec3> bbCog = calculateBoundingBox(positions, qmParticles, bbCutoff);
    // 3nm should be good enough
    filterMMAtomsOMP(positions, mmParticles, bbCog, relevantMMIndices);
    
    std::vector<char> mmRS;
    int numMm = relevantMMIndices.size();

    getSymbolsByIndex(AtomSymbols, relevantMMIndices, mmRS);
    transformPosqMM(positions, charges, relevantMMIndices, mmPos_q);

    // OUTPUT VARIABLES
    std::vector<double> qmForces(numQm * 3, 0), mmForces(numMm * 3, 0);
    std::vector<double> qmQ(numQm, 0);

    double energy;

    Interface.getReaxffPuremdForces(
        numQm, qmSymbols, qmPos, numMm, mmRS, mmPos_q, simBoxInfo, qmForces, mmForces, qmQ, energy);

    // merge the qm and mm forces, additionally transform the scale
    std::vector<Vec3> transformedForces(owner.getNumAtoms(), {0.0, 0.0, 0.0});

    // This is a short operation, no parallelization is needed
    #pragma omp parallel for
    for (size_t i = 0; i < qmParticles.size(); ++i)
    {
        transformedForces[qmParticles[i]][0] = qmForces[3 * i];
        transformedForces[qmParticles[i]][1] = qmForces[3 * i + 1];
        transformedForces[qmParticles[i]][2] = qmForces[3 * i + 2];
    }

    #pragma omp parallel for
    for (size_t i = 0; i < relevantMMIndices.size(); ++i)
    {
        transformedForces[relevantMMIndices[i]][0] = mmForces[i * 3];
        transformedForces[relevantMMIndices[i]][1] = mmForces[i * 3 + 1];
        transformedForces[relevantMMIndices[i]][2] = mmForces[i * 3 + 2];
    }

  
    // copy forces and transform from Hartree/Bohr to kJ/mol/nm

    // is around O(n) so parallelization is useful
    #pragma omp parallel for
    for (size_t i = 0; i < forces.size(); ++i)
    {
        forces[i] = -transformedForces[i] * EvPerNmToKcalPerNm ;
    }

    // done
    // kCal -> kJ and factor...
    return energy * KJPerKcal;
}
