"""Parent class for extracting molecular properties from ASE atoms objects.

Examples
--------
>>> molprops = MolProperties(atoms)
>>> print(type(molprops.geom))
numpy.ndarray
>>> molprops.compute_mass()
>>> molprops.mass
1.67382335232e-27
"""
import functools
from typing import Type, List
import copy

import ase
import ase.io.extxyz
import ase.calculators.calculator
import numpy
import pandas

import ase.io

# Typing
Atoms = Type[ase.Atoms]

class Structure:
    """One structure ar multiple temperatures.

    Parameters
    ----------
    atoms : :obj:`ase.atoms.Atoms`
        The atoms object to conduct analysis on.


    """
    def __init__(self,
                 atoms: Atoms):
        """See class description."""

        # set input
        self.atoms = atoms

        # init computed quantities
        self._mass = None
        self.T = None
        self.log_qpart = None
        return

    @classmethod
    def load_properties(cls,
                        filename: str,
                        csv_filename: str,
                        **kwargs):
        """Construct the class by reading from file.

        Must specify an xyz file containing the geometry of the system, and
        optionally additional parameters to assign. ASE will assume coordinate
        units of Angstrom. Can also specify a csv file containing a table to
        attributes to assign.

        Will only assign attributes loaded from file that exist in the class.

        Parameters
        ----------
        filename : str
            Filepath to file containing geometry to read from.
        csv_filename : str
            Filepath to csv file to read from.
        """
        # first call read geom
        atoms = ase.io.read(filename, format='extxyz')

        # construct the class
        struc = cls(atoms, **kwargs)
        

        prop_table = pandas.read_csv(csv_filename, index_col=0)
        for att in prop_table.columns:
            # if units are appended, or anything else, we need to remove it
            # assume anything after white space is no longer property
            att_ = att.split()[0]
            if att_ == 'T':
                setattr(struc, att_, prop_table[att].values)
            if att_ == 'log_qpart':
                setattr(struc, att_, prop_table[att].values)
        if struc.T is None:
            raise ValueError('No temperatures loaded. Cannot predict. Ensure "T" is a column in the csv file.')
        
        return struc

