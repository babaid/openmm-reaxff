#ifndef OPENMM_REAXFFFORCEIMPL_H
#define OPENMM_REAXFFFORCEIMPL_H
#include "PuremdInterface.h"
#include "openmm/Kernel.h"
#include "openmm/ReaxffForce.h"
#include "openmm/internal/CustomCPPForceImpl.h"

#include <string>
#include <utility>
#include <vector>

namespace OpenMM {

/**
 * This is the internal implementation of ReaxffForce.
 */
class ReaxffForceImpl : public CustomCPPForceImpl
{
  public:
    ReaxffForceImpl(const ReaxffForce &owner);
    ~ReaxffForceImpl() = default;
    double             computeForce(ContextImpl             &context,
                                    const std::vector<Vec3> &positions,
                                    std::vector<Vec3>       &forces) override;
    const ReaxffForce &getOwner() const { return owner; }

  private:
    std::vector<char>   qmSymbols;
    std::vector<char>   AtomSymbols;
    std::vector<int>    qmParticles;
    std::vector<int>    mmParticles;
    std::vector<double> charges;
    const ReaxffForce  &owner;
    PuremdInterface     Interface;

    unsigned int callCounter = 0;
    unsigned int neighborlistUpdateInterval = 10;
    std::vector<int> relevantMMIndices;
};

} // namespace OpenMM
#endif // OPENMM_REAXFFFORCEIMPL_H
