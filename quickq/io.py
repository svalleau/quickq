"""Atoms and MolProps to molml LazyValues"""
from typing import Type
import bidict

import ase
import molml.utils

# patch elements which molml does not have all of
molml.utils.ELE_TO_NUM = bidict.bidict(ase.data.atomic_numbers)

Atoms = Type[ase.Atoms]

def atoms_to_molmllist(atoms: Atoms, bonds: bool = False):
    """Constructs a molml list of lists from an atoms object.
    
    By default, will consider Atomic numbers and atom distances, but can be
    directed to also compute bonds. See the reaxnet.io.rdkit for bond
    determination.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Atoms to compute on.
    bonds : bool, default False
        Whether to compute bonds for the atoms
    """
    positions = atoms.positions
    atomic_numbers = atoms.get_atomic_numbers()
    if bonds:
        raise NotImplementedError(
            'Calculation of molml bond dictionary not implemented yet.'
        )
    else:
        return [atomic_numbers, positions]