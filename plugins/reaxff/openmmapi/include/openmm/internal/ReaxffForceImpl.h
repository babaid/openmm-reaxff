//
// Created by babaid on 05.10.24.
//

#ifndef OPENMM_ReaxffForceImpl_H
#define OPENMM_ReaxffForceImpl_H
#include "openmm/internal/CustomCPPForceImpl.h"
#include "PuremdInterface.h"
#include "openmm/ReaxffForce.h"
#include "openmm/Kernel.h"
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace OpenMM {

/**
 * This is the internal implementation of ReaxffForce.
 */
    class ReaxffForceImpl : public CustomCPPForceImpl {
    public:
      ReaxffForceImpl(const ReaxffForce &owner);
      ~ReaxffForceImpl() = default;
      double computeForce(ContextImpl& context, const std::vector<Vec3> &positions, std::vector<Vec3>& forces) override;
      const ReaxffForce& getOwner() const{
        return owner;
      }
    private:
      std::vector<char> qmSymbols;
      std::vector<char> AtomSymbols;
      std::vector<int> qmParticles;
      std::vector<int> mmParticles;
      std::vector<double> charges;
      const ReaxffForce & owner;
      PuremdInterface Interface;
    };

} // namespace OpenMM
#endif //OPENMM_ReaxffForce_H