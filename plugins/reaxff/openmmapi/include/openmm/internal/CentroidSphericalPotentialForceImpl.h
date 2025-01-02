#ifndef CENTROIDSPHERICALPOTENTIALFORCEIMPL_H
#define CENTROIDSPHERICALPOTENTIALFORCEIMPL_H


#include "openmm/Kernel.h"
#include "openmm/CentroidSphericalPotentialForce.h"
#include "openmm/internal/CustomCPPForceImpl.h"

#include <string>
#include <utility>
#include <vector>

namespace OpenMM {

    /**
     * This is the internal implementation of CentroidSphericalPotentialForce.
     */
    class CentroidSphericalPotentialForceImpl : public CustomCPPForceImpl
    {
    public:
        CentroidSphericalPotentialForceImpl(const CentroidSphericalPotentialForce &owner);
        ~CentroidSphericalPotentialForceImpl() = default;
        double             computeForce(ContextImpl             &context,
                                        const std::vector<Vec3> &positions,
                                        std::vector<Vec3>       &forces) override;
        const CentroidSphericalPotentialForce &getOwner() const { return owner; }

    private:
        std::vector<char>   Atoms;
        double _radius, _strength;
        const CentroidSphericalPotentialForce  &owner;
    };

} // namespace OpenMM

#endif //CENTROIDSPHERICALPOTENTIALFORCEIMPL_H
