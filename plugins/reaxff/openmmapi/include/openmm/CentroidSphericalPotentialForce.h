#ifndef CENTROIDSPHERICALPOTENTIALFORCE_H
#define CENTROIDSPHERICALPOTENTIALFORCE_H

#include "internal/windowsExportReaxff.h"

#include "openmm/Force.h"
#include "openmm/Vec3.h"

#include <map>
#include <vector>

using namespace OpenMM;
namespace OpenMM {

/**
 * A class that introduces a force that creates a spherical potential around the centroid of some atoms.
 */
class OPENMM_EXPORT_REAXFF CentroidSphericalPotentialForce : public Force
{
  public:
    /**
     * Create a CentroidSphericalForce
     *
     */
    CentroidSphericalPotentialForce();
    /**
     * Get the number of atoms being simulated by puremd
     *
     * @return the number of atoms
     */
    int getNumAtoms() const { return Atoms.size(); }
    /**
     * Add a bond term to the force field.
     *
     * @param particle the index of the particle
     * @return the index of the bond that was added
     */
    int addAtom(int particle);
    /**
     * Get the force parameters
     * @param radius the radius of the spherical shell
     * @param strength the strength of the force
     */
    int setForceParameters(double radius, double strength);
    /**
     * Set the force parameters
    * @param radius the radius of the spherical shell
    * @param strength the strength of the force
    */
    void getForceParameters(double& radius, double& strength) const;
    /**
     * Get the bonding atom
     *
     * @param index the index of the atom
     * @param particle the particle index is going to be saved here
     */
    void getParticleParameters(int index, int &particle) const;
    /**
     * Set whether this force should apply periodic boundary conditions when
     * calculating displacements. Usually this is not appropriate for bonded
     * forces, but there are situations when it can be useful.
     */
    void setUsesPeriodicBoundaryConditions(bool periodic);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if force uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const;

  protected:
    ForceImpl *createImpl() const;

  private:
    double _radius, _strength;
    std::vector<int>    Atoms;
    bool                usePeriodic;
    mutable int         numContexts, firstChangedBond, lastChangedBond;
};

} // namespace OpenMM

#endif //CENTROIDSPHERICALPOTENTIALFORCE_H
