#include "openmm/ReaxffForce.h"

#include <cstring>

#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ReaxffForceImpl.h"

using namespace OpenMM;
using namespace std;

ReaxffForce::ReaxffForce(const std::string& ffieldFile,
                         const std::string& controlFile,
                         unsigned int neighborListUpdateInterval) :
    usePeriodic(false),
    numContexts(0),
    ffield_file(ffieldFile),
    control_file(controlFile),
    neighborListUpdateInterval(neighborListUpdateInterval)
{
    this->setName("ReaxFFForce");
}

int ReaxffForce::addAtom(int particle, char* symbol, double charge, bool isQM)
{
    allAtoms.push_back(particle);
    allIsQM.push_back(isQM);
    allSymbols.push_back(symbol[0]);
    allCharges.push_back(charge);
    if (std::strlen(symbol) > 1)
    {
        allSymbols.push_back(symbol[1]);
    }
    else
    {
        allSymbols.push_back('\0');
    }
    return allAtoms.size();
}

void ReaxffForce::addLink(int particle1, int particle2)
{   
    if (particle1 > allAtoms.size() || particle2 > allAtoms.size())
    {
        throw OpenMMException("Particle indices out of bounds.");
    }
    if (allIsQM[particle1] && !allIsQM[particle2])
    {
        linkAtoms.push_back({particle1, particle2});
    }
    else
    {
        throw OpenMMException("Link atom 1 needs to be a ReaxFF atom and link atom 2 needs to be an MM atom.");
    }
}

void ReaxffForce::getLink(int index, int& particle1, int& particle2) const
{
    if (index < linkAtoms.size())
    {
        particle1 = linkAtoms.at(index).first;
        particle2 = linkAtoms.at(index).second;
    }
    else
    {
        throw OpenMMException("List index out of bounds.");
    }
}

void ReaxffForce::getParticleParameters(int index, int& particle, char* symbol,
                                        double& charge, int& isQM) const
{
    particle = allAtoms.at(index);
    symbol[0] = allSymbols.at(index * 2);
    symbol[1] = allSymbols.at(index * 2 + 1);
    charge = allCharges.at(index);
    isQM = allIsQM.at(index);
}

ForceImpl* ReaxffForce::createImpl() const
{
    if (numContexts == 0)
    {
        // Begin tracking changes to atoms.
        firstChangedBond = allAtoms.size();
        lastChangedBond = -1;
    }
    numContexts++;
    return new ReaxffForceImpl(*this);
}

void ReaxffForce::setUsesPeriodicBoundaryConditions(bool periodic)
{
    usePeriodic = periodic;
}

bool ReaxffForce::usesPeriodicBoundaryConditions() const
{
    return usePeriodic;
}
