#include "openmm/ReaxffForce.h"
#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ReaxffForceImpl.h"
#include <cstring>

using namespace OpenMM;
using namespace std;

ReaxffForce::ReaxffForce(const std::string &ffieldFile,
                         const std::string &controlFile, unsigned int neighborListUpdateInterval)
    : usePeriodic(false), numContexts(0), ffield_file(ffieldFile),
      control_file(controlFile), nbUpdateInterval(neighborListUpdateInterval)
{
    this->setName("ReaxFFForce");
}

int ReaxffForce::addAtom(int particle, char *symbol, double charge, bool isQM)
{
    bool hasLen = std::strlen(symbol) > 1;
    allAtoms.push_back(particle);
    allIsQM.push_back(isQM);
    allSymbols.push_back(symbol[0]);
    allCharges.push_back(charge);
    if (hasLen)
    {
        allSymbols.push_back(symbol[1]);
    }
    else
    {
        allSymbols.push_back('\0');
    }
    return allAtoms.size();
}

void ReaxffForce::addLinkAtoms(int particle1, int particle2)
{
    if(allIsQM[particle1] && !allIsQM[particle2] && particle1 < allAtoms.size() && particle2 < allAtoms.size() )
    {
        linkAtoms.push_back({particle1, particle2});
    }
    else
    {
        throw OpenMMException("Link atom can't be added.");
    }
}

void ReaxffForce::getLinkAtoms(int index, int& particle1, int& particle2){
    if (index < linkAtoms.size())
    {
        particle1 = linkAtoms[index].first;
        particle2 = linkAtoms[index].second;
    }
    else{
        throw OpenMMException("List index out of bounds.");
    }
}

void ReaxffForce::getParticleParameters(int index, int &particle, char *symbol,
                                        double &charge, int &isQM) const
{
    particle  = allAtoms[index];
    symbol[0] = allSymbols[index * 2];
    symbol[1] = allSymbols[index * 2 + 1];
    charge    = allCharges[index];
    isQM      = allIsQM[index];
}

ForceImpl *ReaxffForce::createImpl() const
{
    if (numContexts == 0)
    {
        // Begin tracking changes to atoms.
        firstChangedBond = allAtoms.size();
        lastChangedBond  = -1;
    }
    numContexts++;
    return new ReaxffForceImpl(*this);
}

void ReaxffForce::setUsesPeriodicBoundaryConditions(bool periodic)
{
    usePeriodic = periodic;
}

bool ReaxffForce::usesPeriodicBoundaryConditions() const { return usePeriodic; }
