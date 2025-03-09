#include "openmm/internal/ReaxffHelpers.h"

#include<openmm/Units.h>

#include "omp.h"

#include <algorithm>
#include <limits>

using namespace OpenMM;


// flattens the QM atom positions, converts them to angstroms and sets their
// charges  in the context to 0
void ReaxffHelpers::transformPosQM(const std::vector<Vec3> &positions,
                    const std::vector<int> &indices, std::vector<double> &out) {
  out.resize(indices.size() * 3);

#pragma omp parallel for if (indices.size() > parallel_threshold)

  for (size_t i = 0; i < indices.size(); ++i) {
    int Index = indices[i];
    out[i * 3] = positions[indices[i]][0] * AngstromsPerNm;
    out[i * 3 + 1] = positions[indices[i]][1] * AngstromsPerNm;
    out[i * 3 + 2] = positions[indices[i]][2] * AngstromsPerNm;
  }
}

// flattens the MM atom positions converts them to angstroms and converts their
// charges to coulombs
void  ReaxffHelpers::transformPosqMM(const std::vector<Vec3> &positions,
                     const std::vector<double> &charges,
                     const std::vector<int> &indices,
                     std::vector<double> &out) {
  out.resize(indices.size() * 4);

#pragma omp parallel for if (indices.size() > parallel_threshold)
  for (size_t i = 0; i < indices.size(); ++i) {
    out[i * 4] = positions[indices[i]][0] * AngstromsPerNm;
    out[i * 4 + 1] = positions[indices[i]][1] * AngstromsPerNm;
    out[i * 4 + 2] = positions[indices[i]][2] * AngstromsPerNm;
    out[i * 4 + 3] = charges[indices[i]];
  }
}

// filters the all atom symbols list based on indices we want
void  ReaxffHelpers::getSymbolsByIndex(const std::vector<char> &symbols,
                       const std::vector<int> &indices,
                       std::vector<char> &out) {
  out.resize(indices.size() * 2);

#pragma omp parallel for if (indices.size() > parallel_threshold)
  for (size_t i = 0; i < indices.size(); ++i) {
    out[i * 2] = symbols[indices[i] * 2];
    out[i * 2 + 1] = symbols[indices[i] * 2 + 1];
  }
}

// gets the lenghts of the periodic box sides and the angles.
void  ReaxffHelpers::getBoxInfo(const std::vector<Vec3> &positions,
                std::vector<double> &simBoxInfo) {
  // the box will be of the size of the molecule plus a 2nm cutoff.
  double min, max;
  double cutoff = 2.0;

  for (int i = 0; i < 3; i++) {
    min = std::numeric_limits<double>::infinity();
    max = -std::numeric_limits<double>::infinity();
    for (const auto &pos : positions) {
      max = std::max(max, pos[i]);
      min = std::min(min, pos[i]);
    }
    simBoxInfo[i] = (max - min + cutoff) * AngstromsPerNm;
  }
  simBoxInfo[3] = simBoxInfo[4] = simBoxInfo[5] = 90.0;
}

std::pair<Vec3, Vec3>  ReaxffHelpers::calculateBoundingBox(const std::vector<Vec3> &positions,
                                           const std::vector<int> &Indices,
                                           double bbCutoff) {
  Vec3 cutoff = {bbCutoff, bbCutoff, bbCutoff};
  Vec3 minBounds = {std::numeric_limits<double>::max(),
                    std::numeric_limits<double>::max(),
                    std::numeric_limits<double>::max()};
  Vec3 maxBounds = {std::numeric_limits<double>::lowest(),
                    std::numeric_limits<double>::lowest(),
                    std::numeric_limits<double>::lowest()};

  for (const auto &Index : Indices) {
    for (int i = 0; i < 3; ++i) {
      minBounds[i] = std::min(minBounds[i], positions[Index][i]);
      maxBounds[i] = std::max(maxBounds[i], positions[Index][i]);
    }
  }

  return {minBounds - cutoff, maxBounds + cutoff};
}

bool  ReaxffHelpers::isPointInsideBoundingBox(const Vec3 &point,
                              const std::pair<Vec3, Vec3> &boundingBox) {
  // returns false if a point is outside of the bounding box
  const auto &[minBounds, maxBounds] = boundingBox;
  for (int i = 0; i < 3; i++) {
    if ((point[i] < minBounds[i]) || (point[i] > maxBounds[i])) {
      return false;
    }
  }
  return true;
}

// this function is supposed to filter the relevant MM atoms,
// which should considerably speed up the calculations on the PuReMD side
//  This function is O(n)
void  ReaxffHelpers::filterMMAtoms(const std::vector<Vec3> &positions,
                   const std::vector<int> &mmIndices,
                   const std::pair<Vec3, Vec3> &bbCog,
                   std::vector<int> &relevantIndices) {
  const int numThreads = omp_get_num_threads();
  std::vector<std::vector<int>> localIndices(numThreads);

  // Parallel section to filter indices
#pragma omp parallel num_threads(numThreads)
  {
    int threadId = omp_get_thread_num();
    std::vector<int> &localVec = localIndices[threadId];

    // Process each index in mmIndices
#pragma omp for
    for (size_t i = 0; i < mmIndices.size(); i++) {
      const Vec3 &point = positions[mmIndices[i]];
      if ( ReaxffHelpers::isPointInsideBoundingBox(point, bbCog)) {
        localVec.push_back(mmIndices[i]);
      }
    }
  }

  // Merge the results from all threads into relevantIndices
  for (const auto &localVec : localIndices) {
    // Add each local vector's elements to relevantIndices
    relevantIndices.insert(relevantIndices.end(), localVec.begin(),
                           localVec.end());
  }
}

void  ReaxffHelpers::distributeLinkAtomForces(const std::vector<std::pair<int, int>> &linkAtoms,
                              const std::vector<Vec3> &linkAtomPositions,
                              const std::vector<Vec3> &positions,
                              const int numRealQmAtoms,
                              std::vector<double> &qmForces,
                              std::vector<Vec3> &transformedForces) {
  for (size_t i = 0; i < linkAtoms.size(); i++) {
    auto qmAtom = linkAtoms[i].first;
    auto mmAtom = linkAtoms[i].second;

    Vec3 rmqv = positions[mmAtom] - positions[qmAtom];
    Vec3 rlq = linkAtomPositions[i] - positions[qmAtom];
    double rmq = std::sqrt(rmqv.dot(rmqv));

    if (rmq < 1e-8)
      continue;

    Vec3 r_unit = rmqv / rmq;

    Vec3 qmf = {qmForces[(numRealQmAtoms + i) * 3],
                qmForces[(numRealQmAtoms + i) * 3 + 1],
                qmForces[(numRealQmAtoms + i) * 3 + 2]};

    Vec3 proj = r_unit * qmf.dot(r_unit);
    Vec3 F_perp = qmf - proj;
    double scal = std::sqrt(rlq.dot(rlq)) / rmq;
    Vec3 F_mod = scal * F_perp;

    transformedForces[qmAtom] -= (qmf - F_mod);
    transformedForces[mmAtom] -= F_mod;
  }
}

// creates the positions of the link atoms
void  ReaxffHelpers::createLinkAtoms(std::vector<std::pair<int, int>> &linkAtoms,
                     std::vector<Vec3> &linkAtomPositions,
                     const std::vector<Vec3> &positions, double leverFactor) {
  for (int i = 0; i < linkAtoms.size(); i++) {
    auto link = linkAtoms[i];
    Vec3 RL = positions[link.first] +
              leverFactor * (positions[link.second] - positions[link.first]);
    linkAtomPositions[i] = RL;
  }
}

// adds link atom positions and symbols to the qm symbols/positions
void  ReaxffHelpers::addLinkAtoms(const std::vector<std::pair<int, int>> &linkAtoms,
                  const std::vector<Vec3> &linkAtomPositions,
                  std::vector<double> &qmPos, std::vector<char> &qmSymbols) {
  for (int i = 0; i < linkAtoms.size(); i++) {
    qmPos.push_back(linkAtomPositions[i][0]);
    qmPos.push_back(linkAtomPositions[i][1]);
    qmPos.push_back(linkAtomPositions[i][2]);
    qmSymbols.push_back('H');
    qmSymbols.push_back('\0');
  }
}
