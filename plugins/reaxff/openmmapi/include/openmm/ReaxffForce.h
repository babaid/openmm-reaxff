//
// Created by babaid on 05.10.24.
//
#ifndef OPENMM_ReaxffForce_H_
#define OPENMM_ReaxffForce_H_

#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include <map>
#include <vector>
#include "internal/windowsExportReaxff.h"

using namespace OpenMM;
namespace OpenMM {

/**
 * A class that introduces a puremd qmmm force.
 */
    class OPENMM_EXPORT_REAXFF ReaxffForce : public Force {
    public:
    /**
     * Create a puremd force
     */
      ReaxffForce();
    /**
     * Create a ReaxffForce.
     *
     * @param ffieldFile force field file.
     * @param controlFile control file.
     */
      ReaxffForce(const std::string& ffieldfile, const std::string& controlFile);
    /**
     * Get the number of atoms being simulated by puremd
     *
     * @return the number of atoms
     */
    int getNumAtoms() const {
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
     * Get the bonding atom
     *
     * @param index the index of the atoms
     * @param particle the particle index is going to be saved here
     * @param symbol symbol of the atom
     * @param charge charge of the atom
     * @param isQM is it reactive
     */
    void getParticleParameters(int index, int& particle, char* symbol, double& charge, int& isQM) const;
    /**
     * Set whether this force should apply periodic boundary conditions when calculating displacements.
     * Usually this is not appropriate for bonded forces, but there are situations when it can be useful.
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
      std::string ffield_file;
      std::string control_file;
    bool usePeriodic;
    mutable int numContexts, firstChangedBond, lastChangedBond;
};

} // namespace OpenMM


#endif //OPENMM_ReaxffForce_H_