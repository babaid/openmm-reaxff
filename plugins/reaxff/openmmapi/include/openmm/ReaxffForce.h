#ifndef OPENMM_REAXFFFORCE_H_
#define OPENMM_REAXFFFORCE_H_

#include <map>
#include <vector>

#include "internal/windowsExportReaxff.h"
#include "openmm/Force.h"
#include "openmm/Vec3.h"

namespace OpenMM
{
/**
 * A class that introduces a ReaxFF/MM force.
 */
class OPENMM_EXPORT_REAXFF ReaxffForce : public Force
{
   public:
    /**
     * Create a ReaxffForce.
     *
     * @param ffieldFile force field file.
     * @param controlFile control file.
     * @param neighborListUpdateInterval interval of recalculation of which MM
     * atoms are relevant to reaxFF
     */
    ReaxffForce(const std::string& ffieldFile, const std::string& controlFile,
                unsigned int neighborListUpdateInterval);
    /**
     * Get the number of atoms being simulated reactively by puremd
     *
     * @return the number of atoms
     */
    int getNumAtoms() const
    {
        return allAtoms.size();
    }
    /**
     * Gets the filenames used by the force field
     *
     * @param ffieldFile Force field file.
     * @param controlFile Control file.
     */
    void getFileNames(std::string& ffieldFile, std::string& controlFile) const
    {
        ffieldFile = ffield_file;
        controlFile = control_file;
    }
    /**
     * Gets the interval of the neighbor list updates
     *
     * @param Interval interval of recalculation of which MM atoms are relevant
     * to reaxFF
     */
    void getNeighborListUpdateInterval(unsigned int& Interval) const
    {
        Interval = neighborListUpdateInterval;
    }
    /**
     * Add a bond term to the force field.
     *
     * @param particle the index of the particle
     * @param symbol symbol of the particle
     * @param charge charge of the atom
     * @param isQM is it reactive
     * @return the index of the bond that was added
     */
    int addAtom(int particle, char* symbol, double charge, bool isQM);
    /**
     * Creates a link atom representing a covalent bond between a QM atom and an
     * MM atom.
     *
     * @param particle1 The index of the QM atom
     * @param particle2 The index of the MM atom
     */
    void addLink(int particle1, int particle2);
    /**
     * Gets the indices of two linked atoms representing a covalent bond between
     * a QM atom and an MM atom.
     *
     * @param index the index of the link atom pair
     * @param particle1 The index of the QM atom
     * @param particle2 The index of the MM atom
     */
    void getLink(int index, int& particle1, int& particle2) const;
    /**
     * Returns the numbern of link atoms.
     *
     * @return number of link atoms
     */
    int getNumLinks() const
    {
        return linkAtoms.size();
    }
    /**
     * Get information about the atoms simulated by reaxff.
     *
     * @param index the index of the atoms
     * @param particle the particle index is going to be saved here
     * @param symbol symbol of the atom
     * @param charge charge of the atom
     * @param isQM is it reactive
     */
    void getParticleParameters(int index, int& particle, char* symbol,
                               double& charge, int& isQM) const;
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
    ForceImpl* createImpl() const;

   private:
    std::vector<int> allAtoms;
    std::vector<char> allSymbols;
    std::vector<double> allCharges;
    std::vector<bool> allIsQM;
    std::vector<std::pair<int, int>> linkAtoms;
    std::string ffield_file;
    std::string control_file;
    unsigned int neighborListUpdateInterval;
    bool usePeriodic;
    mutable int numContexts, firstChangedBond, lastChangedBond;
};

}  // namespace OpenMM

#endif  // OPENMM_REAXFFFORCE_H_
