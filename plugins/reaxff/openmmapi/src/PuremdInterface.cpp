#include "openmm/internal/PuremdInterface.h"

#include <iostream>

#include "openmm/OpenMMException.h"
#include "spuremd.h"

using namespace OpenMM;

PuremdInterface::PuremdInterface() : firstCall(true)
{
}

PuremdInterface::~PuremdInterface()
{
    retPuremd = cleanup(handlePuremd);
    if (0 != retPuremd)
    {
        throw OpenMMException("Error at cleanup in PuReMD.");
    }
}

void PuremdInterface::setInputFileNames(const std::string& ffieldFilename,
                                        const std::string& controlFilename)
{
    ffield_filename = ffieldFilename;
    control_filename = controlFilename;
}

void PuremdInterface::getReaxffPuremdForces(
    int num_qm_atoms, const std::vector<char>& qm_symbols,
    const std::vector<double>& qm_pos, int num_mm_atoms,
    const std::vector<char>& mm_symbols, const std::vector<double>& mm_pos_q,
    const std::vector<double>& sim_box_info, std::vector<double>& qm_forces,
    std::vector<double>& mm_forces, std::vector<double>& qm_q,
    double& totalEnergy)
{
    if (firstCall)
    {
        handlePuremd = setup_qmmm(
            num_qm_atoms, qm_symbols.data(), qm_pos.data(), num_mm_atoms,
            mm_symbols.data(), mm_pos_q.data(), sim_box_info.data(),
            ffield_filename.c_str(), control_filename.c_str());
        firstCall = false;
    }
    else
    {
        retPuremd =
            reset_qmmm(handlePuremd, num_qm_atoms, qm_symbols.data(),
                       qm_pos.data(), num_mm_atoms, mm_symbols.data(),
                       mm_pos_q.data(), sim_box_info.data(),
                       ffield_filename.c_str(), control_filename.c_str());
        if (0 != retPuremd) throw OpenMMException("Error at reset PuReMD.");
    }

    retPuremd = simulate(handlePuremd);

    if (0 != retPuremd) throw OpenMMException("Error at PuReMD simulation.");
    retPuremd =
        get_atom_forces_qmmm(handlePuremd, qm_forces.data(), mm_forces.data());
    if (0 != retPuremd)
        throw OpenMMException("Error retrieving forces from PuReMD.");
    retPuremd = get_atom_charges_qmmm(handlePuremd, qm_q.data(), NULL);
    if (0 != retPuremd)
        throw OpenMMException("Error retrieving charges from PuReMD.");
    retPuremd = get_system_info(handlePuremd, NULL, NULL, &totalEnergy, NULL,
                                NULL, NULL);
    if (0 != retPuremd)
        throw OpenMMException("Error retrieving energy from PuReMD.");
}
