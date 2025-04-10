#ifndef OPENMM_REAXFFHELPERS_H
#define OPENMM_REAXFFHELPERS_H

#include <openmm/Vec3.h>

#include <vector>

using namespace OpenMM;

namespace ReaxffHelpers
{

constexpr size_t parallel_threshold = 100;
/**
 * Extract the positions of the QM atoms from the list of 
 * particles, and convert the 3D vectors to 1D, from nanometers to angstroms.
 */
void transformPosQM(const std::vector<Vec3>& positions,
                    const std::vector<int>& indices, std::vector<double>& out);

/**
 * Extract the positions and the charges of the MM atoms from the list of 
 * particles, and convert the 3D vectors to 1D, from nanometers to angstroms.
 */
void transformPosqMM(const std::vector<Vec3>& positions,
                     const std::vector<double>& charges,
                     const std::vector<int>& indices, std::vector<double>& out);

/**
 * Return the symbols of some atoms given by a vector of particle indices.
 */
void getSymbolsByIndex(const std::vector<char>& symbols,
                       const std::vector<int>& indices, std::vector<char>& out);

/**
 * Gets the box dimensions (in angstroms) based on the QM atom positions, needed by PuReMD.
 */
void getBoxInfo(const std::vector<Vec3>& positions,
                std::vector<double>& simBoxInfo);

/**
 * Calculates a bounding around the QM atoms, with each edge having a 1nm offset.
 * Needed to calculate neighbor lists. 
 */
std::pair<Vec3, Vec3> calculateBoundingBox(const std::vector<Vec3>& positions,
                                           const std::vector<int>& Indices,
                                           double bbCutoff);
/** 
 * Checks if a point/particle is inside of a bounding spanned by two distinct points in space.
*/
bool isPointInsideBoundingBox(const Vec3& point,
                              const std::pair<Vec3, Vec3>& boundingBox);

/**
 * Gets the truly relevant MM atoms within the bounding box.
 *  */                              
void filterMMAtoms(const std::vector<Vec3>& positions,
                   const std::vector<int>& mmIndices,
                   const std::pair<Vec3, Vec3>& bbCog,
                   std::vector<int>& relevantIndices);
/**
 * Distributes the link atom forces among the 
 * respective QM and MM atom pairs using the lever rule.
 *  */                   
void distributeLinkAtomForces(const std::vector<std::pair<int, int>>& linkAtoms,
                              const std::vector<Vec3>& linkAtomPositions,
                              const std::vector<Vec3>& positions,
                              const int numRealQmAtoms,
                              std::vector<double>& qmForces,
                              std::vector<Vec3>& transformedForces);
/**
* Populates the positions with the new link atom positions.
*/
void createLinkAtoms(std::vector<std::pair<int, int>>& linkAtoms,
                     std::vector<Vec3>& linkAtomPositions,
                     const std::vector<Vec3>& positions, double leverFactor);
/**
 * Adds the virtual link atom positions and symbols to the respective QM vectors.
 */
void addLinkAtoms(const std::vector<std::pair<int, int>>& linkAtoms,
                  const std::vector<Vec3>& linkAtomPositions,
                  std::vector<double>& qmPos, std::vector<char>& qmSymbols);

void reshapeFlattenedForces(std::vector<Vec3>& transformedForces, const std::vector<int>& indices, const std::vector<double> flattenedForces);

void finalizeForceConversion(std::vector<Vec3>& forces, const std::vector<Vec3>& transformedForces);

}  // namespace ReaxffHelpers

#endif