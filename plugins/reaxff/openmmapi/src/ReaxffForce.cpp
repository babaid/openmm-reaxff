//
// Created by babaid on 05.10.24.
//

#include "openmm/ReaxffForce.h"
#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ReaxffForceImpl.h"
#include <cstring>
using namespace OpenMM;
using namespace std;

ReaxffForce::ReaxffForce() {}

ReaxffForce::ReaxffForce(const std::string& ffieldFile, const std::string& controlFile) :usePeriodic(false), numContexts(0), ffield_file(ffieldFile), control_file(controlFile) {
}

int ReaxffForce::addAtom(int particle, char* symbol, double charge, bool isQM) {
    bool hasLen = std::strlen(symbol) > 1;
    allAtoms.push_back(particle);
    allIsQM.push_back(isQM);
    allSymbols.push_back(symbol[0]);
    allCharges.push_back(charge);
    if (hasLen){
      allSymbols.push_back(symbol[1]);
    }
    else
    {
      allSymbols.push_back('\0');
    }
    return allAtoms.size();
}

void ReaxffForce::getParticleParameters(int index, int &particle, char* symbol, double& charge, int &isQM)  const {
    particle = allAtoms[index];
    symbol[0] = allSymbols[index*2];
    symbol[1] = allSymbols[index*2 + 1];
    charge = allCharges[index];
    isQM = allIsQM[index];
}

ForceImpl*ReaxffForce::createImpl() const {
    if (numContexts == 0) {
        // Begin tracking changes to atoms.
        firstChangedBond = allAtoms.size();
        lastChangedBond = -1;
    }
    numContexts++;
    return new ReaxffForceImpl(*this);
}

void ReaxffForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

bool ReaxffForce::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}

