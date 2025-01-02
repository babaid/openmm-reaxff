#include "openmm/CentroidSphericalPotentialForce.h"
#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/CentroidSphericalPotentialForceImpl.h"
#include <cstring>

using namespace OpenMM;
using namespace std;

CentroidSphericalPotentialForce::CentroidSphericalPotentialForce()
    : usePeriodic(false), numContexts(0)
{
}

int CentroidSphericalPotentialForce::addAtom(int particle)
{
    Atoms.push_back(particle);
    return Atoms.size();
}

int CentroidSphericalPotentialForce::setForceParameters(double radius, double strength)
{
    _radius = radius;
    _strength = strength;
    return 0;
}

void CentroidSphericalPotentialForce::getForceParameters(double& radius, double& strength) const
{
    radius = _radius;
    strength = _strength;
}

void CentroidSphericalPotentialForce::getParticleParameters(int index, int &particle) const
{
    particle  = Atoms[index];
}

ForceImpl *CentroidSphericalPotentialForce::createImpl() const
{
    if (numContexts == 0)
    {
        // Begin tracking changes to atoms.
        firstChangedBond = Atoms.size();
        lastChangedBond  = -1;
    }
    numContexts++;
    return new CentroidSphericalPotentialForceImpl(*this);
}

void CentroidSphericalPotentialForce::setUsesPeriodicBoundaryConditions(bool periodic)
{
    usePeriodic = periodic;
}

bool CentroidSphericalPotentialForce::usesPeriodicBoundaryConditions() const { return usePeriodic; }
