#include "openmm/internal/CentroidSphericalPotentialForceImpl.h"

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


CentroidSphericalPotentialForceImpl::CentroidSphericalPotentialForceImpl(const CentroidSphericalPotentialForce &owner)
    : CustomCPPForceImpl(owner), owner(owner)
{
    double radius;
    double strength;
    owner.getForceParameters(radius, strength);
    _radius = radius;
    _strength = strength;
    for (int i = 0; i < owner.getNumAtoms(); ++i)
    {
        int    particle;
        owner.getParticleParameters(i, particle);
        Atoms.push_back(particle);
    }
}
double CentroidSphericalPotentialForceImpl::computeForce(ContextImpl             &context,
                                     const std::vector<Vec3> &positions,
                                     std::vector<Vec3>       &forces)
{
    int numAtoms = Atoms.size();
    Vec3 Centroid{0.0, 0.0, 0.0};
    size_t Index;
    #pragma omp parallel for
    for (size_t i=0; i< numAtoms; i++)
    {
        Index = Atoms[i];
        Centroid+=positions[Index];
    }

    Centroid/=numAtoms;

    double distance;
    double energy = 0.0;
    Vec3 U;

    #pragma parallel for
    for (size_t i = 0; i<positions.size(); i++) forces[i] = Vec3(0.0,0.0,0.0);

    #pragma omp barrier

    double dE_dr;
    double threadEnergy = 0.0;
    #pragma omp parallel for reduction(+:threadEnergy)
    for (size_t i = 0; i < numAtoms; i++)
    {   
        Index = Atoms[i];
        U = positions[Index] - Centroid;
        distance = std::sqrt(U.dot(U));
        dE_dr = std::max(0.0, distance - _radius);
        threadEnergy += 0.5 * _strength * std::pow(dE_dr, 2);
        if(distance != 0.0)
        {
        forces[Index][0] = - _strength * dE_dr *  U[0] / distance;
        forces[Index][1] = - _strength * dE_dr *  U[1] / distance;
        forces[Index][2] = - _strength * dE_dr *  U[2] / distance;
        }
    }
    energy = threadEnergy;
    
    return energy;
}
