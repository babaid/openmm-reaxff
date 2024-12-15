import numpy as np
from openmm.app import Modeller
from openmm import *

def is_in_molecule(testatoms, molatoms):
    for atom in testatoms:
        if atom in molatoms:
            return True
    return False


def remove_extra_forces(system, reax_atoms):
    harmonic_bond_force = None
    angle_force = None
    torsion_force = None

    protein_force = HarmonicBondForce()
    protein_angle_force = HarmonicAngleForce()
    protein_torsion_force = PeriodicTorsionForce()

    for n, force in enumerate(system.getForces()):
        if isinstance(force, HarmonicBondForce):
            harmonic_bond_force = force
            print("Number of bond forces before update: ", harmonic_bond_force.getNumBonds())
            for i in range(harmonic_bond_force.getNumBonds()):
                atom1, atom2, length, k = harmonic_bond_force.getBondParameters(i)
                if is_in_molecule([atom1, atom2], reax_atoms):
                    continue
                else:
                    protein_force.addBond(atom1, atom2, length, k)
            print("Number of bond forces after update: ", protein_force.getNumBonds())
            system.removeForce(n)
            break
            
    for n, force in enumerate(system.getForces()):
        if isinstance(force, HarmonicAngleForce):
            angle_force = force
            
            print("Number of angle forces before update: ", angle_force.getNumAngles())
            for i in range(angle_force.getNumAngles()):
                atom1, atom2, atom3, angle, k = angle_force.getAngleParameters(i)
                if is_in_molecule([atom1, atom2, atom3], reax_atoms):
                    continue
                else:
                    protein_angle_force.addAngle(atom1, atom2, atom3, angle, k)
            print("Number of angle forces after update: ", protein_angle_force.getNumAngles())
            system.removeForce(n)
            break
            
    for n, force in enumerate(system.getForces()):
        if isinstance(force, PeriodicTorsionForce):
            torsion_force = force
            print("Number of torsion forces before update: ", torsion_force.getNumTorsions())
            for i in range(torsion_force.getNumTorsions()):
                atom1, atom2, atom3, atom4, periodicity, phase, k = torsion_force.getTorsionParameters(i)
                if is_in_molecule(  [atom1, atom2, atom3, atom4], reax_atoms):  
                    continue
                else:
                    protein_torsion_force.addTorsion(atom1, atom2, atom3, atom4, periodicity, phase, k)
            print("Number of torsion forces after update: ", protein_torsion_force.getNumTorsions())
            system.removeForce(n)
            break    
    system.addForce(protein_angle_force)
    system.addForce(protein_force)
    system.addForce(protein_torsion_force)