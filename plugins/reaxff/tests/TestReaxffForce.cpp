#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/ReaxffForce.h"
#include "openmm/System.h"
#include "openmm/Units.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/AssertionUtilities.h"
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

const double TOL = 1e-5;

#include <cstdlib>

std::string getAbsolutePath(const std::string &relativePath)
{
#ifdef _WIN32
    char absPath[_MAX_PATH];
    if (!_fullpath(absPath, relativePath.c_str(), _MAX_PATH))
    {
        throw std::runtime_error("Error resolving path: " + relativePath);
    }
    return std::string(absPath);
#else
    char *absPath = realpath(relativePath.c_str(), nullptr);
    if (!absPath)
    {
        throw std::runtime_error("Error resolving path: " + relativePath);
    }
    std::string result(absPath);
    free(absPath);
    return result;
#endif
}

const std::string CONTROL = "../plugins/reaxff/tests/ffield.reaxff";
const std::string FFIELD  = "../plugins/reaxff/tests/control";

Platform &platform = Platform::getPlatformByName("Reference");

int validateForce()
{
    System system;

    system.addParticle(16.0);
    system.addParticle(16.0);
    VerletIntegrator integrator(0.01);
    ReaxffForce     *forceField = new ReaxffForce(CONTROL, FFIELD);

    forceField->addAtom(0, "O", 0.0, true);
    forceField->addAtom(1, "O", 0.0, true);

    system.addForce(forceField);
    Context context(system, integrator, platform);

    vector<Vec3> positions(2);

    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 0.116, 0);

    context.setPositions(positions);

    {
        integrator.step(100);
        State state = context.getState(State::Forces | State::Energy);
        const vector<Vec3> &forces = state.getForces();
        const double        energy = state.getPotentialEnergy();

        std::cout << energy << std::endl;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    try
    {
        validateForce();
    }
    catch (const exception &e)
    {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}