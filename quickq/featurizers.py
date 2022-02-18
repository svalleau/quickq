"""Featurizers of systems represented by Structures.
"""
from typing import Union, Type, Iterable

import ase
import molml.features
import numpy

import quickq.structure
import quickq.io
# Typing
Structure = Type[quickq.structure.Structure]

class MolFeaturizer:
    """Parent class for featurizing structures.
    
    Attributes
    ----------
    name : str
        Name of featurizer
    """
    name = 'MolFeaturizer'
    def __init__(self, **kwargs):
        return
    
    def featurize(self, systems: Union[Structure, Iterable[Structure]], **kwargs) -> list:
        """Featurize a set of Structures.
        
        Parameters
        ----------
        systems : Iterable of Structure
            The systems to featurize
            
        Returns
        -------
        list of numpy.ndarray
        """
        # first handle types, we want an iterable
        if type(systems) == list:
            systems_ = systems
        elif type(systems) == numpy.ndarray:
            systems_ = systems
        else:
            systems_ = [systems]
            
        # now get ase or Structure
        if all(
            [
                isinstance(
                    system, quickq.structure.Structure
                ) for system in systems_
            ]
        ):
            pass
        else:
            raise ValueError(
                'systems must all be Structure'
            )
        
        # do the featurization
        features = self._featurize(systems_, **kwargs)
        if len(features) != len(systems_):
            raise ValueError(
                'Expected list of features same length as the systems input.'
            )
        features = numpy.array(features)
        return features
    
    def _featurize(self, systems_: Iterable[Structure], **kwargs) -> list:
        """Must be overloaded by child classes."""
        raise NotImplementedError(
            f'`_features` method not defined for featurizer {self.name}')
        return None
      
class MolMLFeaturizer(MolFeaturizer):
    """Featurizer wrapping any molml featurizer.
    
    Parameters
    ----------
    name : str
        Name of featurizer in molml
    bonds : bool
        whether to consider bonds when computing features
    fit_atoms : list of ase.Atoms
        Atoms to be used for fitting featurizer before computing the entire
        dataset. Molml does not allow specification of maximums for sustem
        size dependant features, so must be determined by fitting to the
        largest and most diverse systems.
    **kwargs passed to molml featurizer construction
    
    Attributes
    ----------
    name : str
        Name of featurizer
    """
    name = None
    def __init__(
        self,
        name,
        bonds: bool = False,
        fit_atoms: list = None,
        **kwargs
    ):
        super().__init__()
        self.mmlFeaturizer = getattr(molml.features, name)(**kwargs)
        self.name = name
        self.bonds = bonds
        if fit_atoms is not None:
            self._fit(fit_atoms)
        return
    
    def _fit(self, fit_atoms: list):
        """Fit the transfomer ahead of time to fit atoms"""
        molml_lists = [
            quickq.io.atoms_to_molmllist(
                    atoms, bonds=self.bonds
            ) for atoms in fit_atoms
        ]
        self.mmlFeaturizer.fit(molml_lists)
        return
    
    def _featurize(self, systems_: Iterable[Structure], **kwargs) -> list:
        molml_lists = [
            quickq.io.atoms_to_molmllist(
                    system.atoms, bonds=self.bonds
            ) for system in systems_
        ]
        # if fit wasn't done on fit atoms do a fit transform
        # dangerous if any shard of a dataset has different maximums 
        # from another
        try: 
            features = self.mmlFeaturizer.transform(molml_lists, **kwargs)
        except ValueError:
            features = self.mmlFeaturizer.fit_transform(molml_lists, **kwargs)
        return features
        
