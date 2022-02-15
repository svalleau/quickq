"""Featurizers of systems represented by MolProperties.
"""
from typing import Union, Type, Iterable

import ase
import deepchem.feat
import dscribe.descriptors
import molml.features
import numpy

import quickq.structure
import quickq.io
# Typing
Structure = Type[quickq.structure.Structure]

class MolFeaturizer:
    """Parent class for featurizes molecules.
    
    Parameters
    ----------
    store : bool
        Whether or not to store features in Structure object as well as return
        the vector.
    
    Attributes
    ----------
    name : str
        Name of featurizer
    """
    name = 'MolFeaturizer'
    def __init__(self, store: bool = False, **kwargs):
        if type(store) != bool:
            raise ValueError('store should be a bool')
        return
    
    def featurize(self, systems: Union[Structure, Iterable[Structure]], **kwargs) -> list:
        """Featurize a set of MolProp systems.
        
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
    

class DscribeFeaturizer(MolFeaturizer):
    """Featurizer wrapping any dscribe featurizer.
    
    Parameters
    ----------
    name : str
        Name of featurizer in dscribe
    store : bool
        Whether or not to store features in Structure object as well as return
        the vector.
    **kwargs passed to dscribe featurizer construction
    
    Attributes
    ----------
    name : str
        Name of featurizer
    """
    name = None
    def __init__(self, name: str, store: bool = False, **kwargs):
        super().__init__(store)
        self.dsFeaturizer = getattr(dscribe.descriptors, name)(**kwargs)
        self.name = name
        return
    
    def _featurize(self, systems_: Iterable[Structure], **kwargs) -> list:
        atomlist = [system.atoms for system in systems_]
        features = self.dsFeaturizer.create(atomlist, **kwargs)
        return features
    
class MolMLFeaturizer(MolFeaturizer):
    """Featurizer wrapping any molml featurizer.
    
    Parameters
    ----------
    name : str
        Name of featurizer in dscribe
    store : bool
        Whether or not to store features in Structure object as well as return
        the vector.
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
        
