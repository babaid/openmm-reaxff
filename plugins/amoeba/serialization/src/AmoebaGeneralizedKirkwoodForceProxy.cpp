/* -------------------------------------------------------------------------- *
 *                                OpenMMAmoeba                                *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010-2016 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/serialization/AmoebaGeneralizedKirkwoodForceProxy.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/Force.h"
#include "openmm/AmoebaGeneralizedKirkwoodForce.h"
#include <sstream>

using namespace OpenMM;
using namespace std;

AmoebaGeneralizedKirkwoodForceProxy::AmoebaGeneralizedKirkwoodForceProxy() : SerializationProxy("AmoebaGeneralizedKirkwoodForce") {
}

void AmoebaGeneralizedKirkwoodForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 3);
    const AmoebaGeneralizedKirkwoodForce& force = *reinterpret_cast<const AmoebaGeneralizedKirkwoodForce*>(object);

    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    node.setDoubleProperty("GeneralizedKirkwoodSolventDielectric", force.getSolventDielectric());
    node.setDoubleProperty("GeneralizedKirkwoodSoluteDielectric",  force.getSoluteDielectric());
    node.setDoubleProperty("GeneralizedKirkwoodDielectricOffset",  force.getDielectricOffset());
    node.setDoubleProperty("GeneralizedKirkwoodProbeRadius",       force.getProbeRadius());
    node.setDoubleProperty("GeneralizedKirkwoodSurfaceAreaFactor", force.getSurfaceAreaFactor());
    node.setIntProperty(  "GeneralizedKirkwoodIncludeCavityTerm", force.getIncludeCavityTerm());
    node.setBoolProperty("GeneralizedKirkwoodTanhRescaling", force.getTanhRescaling());
    double b0, b1, b2;
    force.getTanhParameters(b0, b1, b2);
    node.setDoubleProperty("GeneralizedKirkwoodTanhB0", b0);
    node.setDoubleProperty("GeneralizedKirkwoodTanhB1", b1);
    node.setDoubleProperty("GeneralizedKirkwoodTanhB2", b2);
    node.setDoubleProperty("GeneralizedKirkwoodDescreenOffset", force.getDescreenOffset());
    SerializationNode& particles = node.createChildNode("GeneralizedKirkwoodParticles");
    for (unsigned int ii = 0; ii < static_cast<unsigned int>(force.getNumParticles()); ii++) {
        double radius, charge, scalingFactor, descreenRadius, neckFactor;
        force.getParticleParameters(ii, charge, radius, scalingFactor, descreenRadius, neckFactor);
        particles.createChildNode("Particle").setDoubleProperty("charge", charge).setDoubleProperty("radius", radius).setDoubleProperty("scaleFactor", scalingFactor).setDoubleProperty("descreenRadius", descreenRadius).setDoubleProperty("neckFactor", neckFactor);
    }

}

void* AmoebaGeneralizedKirkwoodForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version < 1 || version > 3)
        throw OpenMMException("Unsupported version number");
    AmoebaGeneralizedKirkwoodForce* force = new AmoebaGeneralizedKirkwoodForce();
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setSolventDielectric(node.getDoubleProperty("GeneralizedKirkwoodSolventDielectric"));
        force->setSoluteDielectric(node.getDoubleProperty("GeneralizedKirkwoodSoluteDielectric"));
        force->setProbeRadius(node.getDoubleProperty("GeneralizedKirkwoodProbeRadius"));
        force->setSurfaceAreaFactor(node.getDoubleProperty("GeneralizedKirkwoodSurfaceAreaFactor"));
        force->setIncludeCavityTerm(node.getIntProperty("GeneralizedKirkwoodIncludeCavityTerm"));
        if (version > 2) {
            force->setDielectricOffset(node.getDoubleProperty("GeneralizedKirkwoodDielectricOffset"));
            force->setTanhRescaling(node.getBoolProperty("GeneralizedKirkwoodTanhRescaling"));
            double b0 = node.getDoubleProperty("GeneralizedKirkwoodTanhB0");
            double b1 = node.getDoubleProperty("GeneralizedKirkwoodTanhB1");
            double b2 = node.getDoubleProperty("GeneralizedKirkwoodTanhB2");
            force->setTanhParameters(b0, b1, b2);
            force->setDescreenOffset(node.getDoubleProperty("GeneralizedKirkwoodDescreenOffset"));

        }

        const SerializationNode& particles = node.getChildNode("GeneralizedKirkwoodParticles");
        for (unsigned int ii = 0; ii < particles.getChildren().size(); ii++) {
            const SerializationNode& particle = particles.getChildren()[ii];
            double charge = particle.getDoubleProperty("charge");
            double radius = particle.getDoubleProperty("radius");
            double scaleFactor = particle.getDoubleProperty("scaleFactor");
            double descreenRadius = radius;
            double neckFactor = 0.0;
            if (version > 2) {
                descreenRadius = particle.getDoubleProperty("descreenRadius");
                neckFactor = particle.getDoubleProperty("neckFactor");
            }
            force->addParticle(charge, radius, scaleFactor, descreenRadius, neckFactor);
        }
    }
    catch (...) {
        delete force;
        throw;
    }

    return force;
}
