#ifndef OPENMM_REAXFFHELPERS_H
#define OPENMM_REAXFFHELPERS_H

#include <openmm/Vec3.h>
#include <vector>

using namespace OpenMM;

namespace ReaxffHelpers {

    constexpr size_t parallel_threshold = 100;
    
    void transformPosQM(const std::vector<Vec3> &positions,
                        const std::vector<int> &indices, std::vector<double> &out);

    void transformPosqMM(const std::vector<Vec3> &positions,
                        const std::vector<double> &charges,
                        const std::vector<int> &indices, std::vector<double> &out);

    void getSymbolsByIndex(const std::vector<char> &symbols,
                        const std::vector<int> &indices, std::vector<char> &out);

    void getBoxInfo(const std::vector<Vec3> &positions,
                    std::vector<double> &simBoxInfo);

    std::pair<Vec3, Vec3> calculateBoundingBox(const std::vector<Vec3> &positions,
                                            const std::vector<int> &Indices,
                                            double bbCutoff);

    bool isPointInsideBoundingBox(const Vec3 &point,
                                const std::pair<Vec3, Vec3> &boundingBox);

    void filterMMAtoms(const std::vector<Vec3> &positions,
                    const std::vector<int> &mmIndices,
                    const std::pair<Vec3, Vec3> &bbCog,
                    std::vector<int> &relevantIndices);

    void distributeLinkAtomForces(const std::vector<std::pair<int, int>> &linkAtoms,
                                const std::vector<Vec3> &linkAtomPositions,
                                const std::vector<Vec3> &positions,
                                const int numRealQmAtoms,
                                std::vector<double> &qmForces,
                                std::vector<Vec3> &transformedForces);

    void createLinkAtoms(std::vector<std::pair<int, int>> &linkAtoms,
                            std::vector<Vec3> &linkAtomPositions,
                            const std::vector<Vec3> &positions,
                            double leverFactor);

    void addLinkAtoms(const std::vector<std::pair<int, int>> &linkAtoms,
                    const std::vector<Vec3> &linkAtomPositions,
                    std::vector<double> &qmPos, std::vector<char> &qmSymbols);

} // namespace ReaxffHelpers

#endif