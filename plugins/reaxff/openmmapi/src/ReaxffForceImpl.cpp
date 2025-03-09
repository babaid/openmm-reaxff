#include "openmm/internal/ReaxffHelpers.h"

#include "openmm/internal/PuremdInterface.h"
#include "openmm/internal/ReaxffForceImpl.h"

#include "openmm/OpenMMException.h"
#include "openmm/Units.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/kernels.h"

#include "omp.h"

#include <algorithm>
#include <limits>

using namespace ReaxffHelpers;
using namespace OpenMM;

ReaxffForceImpl::ReaxffForceImpl(const ReaxffForce &owner)
    : CustomCPPForceImpl(owner), owner(owner) {
  std::string ffield_file, control_file;
  owner.getFileNames(ffield_file, control_file);
  Interface.setInputFileNames(ffield_file, control_file);
  owner.getNeighborListUpdateInterval(neighborlistUpdateInterval);

  for (int i = 0; i < owner.getNumAtoms(); ++i) {
    int particle;
    char symbol[2];
    int isqm;
    double charge;
    owner.getParticleParameters(i, particle, symbol, charge, isqm);
    if (isqm) {
      qmParticles.emplace_back(particle);
      qmSymbols.emplace_back(symbol[0]);
      qmSymbols.emplace_back(symbol[1]);
    } else {
      mmParticles.emplace_back(particle);
    }
    AtomSymbols.emplace_back(symbol[0]);
    AtomSymbols.emplace_back(symbol[1]);
    charges.emplace_back(charge);
  }
  for (int i = 0; i < owner.getNumLinks(); i++) {
    int particle1, particle2;
    owner.getLink(i, particle1, particle2);
    linkAtoms.push_back({particle1, particle2});
    linkAtomPositions.push_back(Vec3(0.0, 0.0, 0.0));
  }
}
double ReaxffForceImpl::computeForce(ContextImpl &context,
                                     const std::vector<Vec3> &positions,
                                     std::vector<Vec3> &forces) {
  int N = owner.getNumAtoms();
  int numQm = qmParticles.size() + linkAtoms.size();

  std::vector<double> qmPos, mmPos_q;
  std::vector<double> simBoxInfo(6);
  
  // set up link atoms...
  createLinkAtoms(linkAtoms, linkAtomPositions, positions, 0.723);

  // box info for reaxff
  getBoxInfo(positions, simBoxInfo);

  // split QM atom positions and MM atom positions + charges
  transformPosQM(positions, qmParticles, qmPos);

  addLinkAtoms(linkAtoms, linkAtomPositions, qmPos, qmSymbols);

  // get relevant MM indices from a bounding box sorrounding the ReaxFF atoms
  //  ~1nm makes sense as it is the upper taper radius in the ReaxFF
  //  nonbondonded potentials
  double bbCutoff = 1.0;
  std::pair<Vec3, Vec3> bbCog =
      calculateBoundingBox(positions, qmParticles, bbCutoff);

  // std::vector<int> relevantMMIndices;
  if (callCounter % neighborlistUpdateInterval == 0) {
    relevantMMIndices.clear();
    filterMMAtoms(positions, mmParticles, bbCog, relevantMMIndices);
  }

  std::vector<char> mmAtomSymbols;
  int numMMAtoms = relevantMMIndices.size();

  getSymbolsByIndex(AtomSymbols, relevantMMIndices, mmAtomSymbols);
  transformPosqMM(positions, charges, relevantMMIndices, mmPos_q);
  // OUTPUT
  std::vector<double> qmForces(numQm * 3, 0), mmForces(numMMAtoms * 3, 0);
  std::vector<double> qmQ(numQm, 0);
  double energy;

  Interface.getReaxffPuremdForces(numQm, qmSymbols, qmPos, numMMAtoms,
                                  mmAtomSymbols, mmPos_q, simBoxInfo, qmForces,
                                  mmForces, qmQ, energy);
  // Merge QM and MM forces
  std::vector<Vec3> transformedForces(owner.getNumAtoms(), {0.0, 0.0, 0.0});

#pragma omp parallel for if (qmParticles.size() > parallel_threshold)
  for (size_t i = 0; i < qmParticles.size(); ++i) {
    transformedForces[qmParticles[i]][0] = qmForces[3 * i];
    transformedForces[qmParticles[i]][1] = qmForces[3 * i + 1];
    transformedForces[qmParticles[i]][2] = qmForces[3 * i + 2];
  }

#pragma omp parallel for if (relevantMMIndices.size() > parallel_threshold)
  for (size_t i = 0; i < relevantMMIndices.size(); ++i) {
    transformedForces[relevantMMIndices[i]][0] = mmForces[i * 3];
    transformedForces[relevantMMIndices[i]][1] = mmForces[i * 3 + 1];
    transformedForces[relevantMMIndices[i]][2] = mmForces[i * 3 + 2];
  }

  // distribute forces of link atoms between qm and mm atom using lever rule.
  distributeLinkAtomForces(linkAtoms, linkAtomPositions, positions,
                           qmParticles.size(), qmForces, transformedForces);

#pragma omp parallel for if (AtomSymbols.size() > parallel_threshold)
  for (size_t i = 0; i < forces.size(); ++i) {
    forces[i] = -transformedForces[i] * AngstromsPerNm * KJPerKcal * 0.1;
  }
  callCounter++;
  return energy * KJPerKcal * 0.1;
}
