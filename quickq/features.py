"""Defines the featurizers to use in the project.

MolML requires the systems themselves to fit the featurizer. This is a problem
when we featurize individual chunks at a time. We must first retrieve the fit
atoms found using find_biggest_systems.py.

dscribe requires explicit definition of parameters, done below.
"""

import ase.io
from quickq.featurizers import DscribeFeaturizer, MolMLFeaturizer
import sys
import os
filepath = os.path.dirname(os.path.realpath(__file__))
fit_atoms_dir = os.path.join(filepath, 'fit_atoms')

# load the fit atoms for molml featurizers
fit_atoms = [ase.io.read(fit_atoms_dir+'/'+system) for system in os.listdir(fit_atoms_dir) if system.endswith('xyz')]

# define each featurizer
featurizers = {}
SOAP = DscribeFeaturizer(**{
    'name': 'SOAP',
    'species': ["H", "C", "O", "N"],
    'rcut': 6.0,
    'nmax': 8,
    'lmax': 6,
    'average': 'inner'
})
featurizers['SOAP'] = SOAP

MBTR = DscribeFeaturizer(
    name='MBTR',
    species=["H", "O", 'C', 'N'],
    k1={
        "geometry": {"function": "atomic_number"},
        "grid": {"min": 0, "max": 8, "n": 100, "sigma": 0.1},
    },
    k2={
        "geometry": {"function": "inverse_distance"},
        "grid": {"min": 0, "max": 1, "n": 100, "sigma": 0.1},
        "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
    },
    k3={
        "geometry": {"function": "cosine"},
        "grid": {"min": -1, "max": 1, "n": 100, "sigma": 0.1},
        "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
    },
    periodic=False,
    normalization='none',
    flatten=True
)
featurizers['MBTR'] = MBTR

EncodedBonds = MolMLFeaturizer(
    name='EncodedBond',
    fit_atoms=fit_atoms,
    add_unknown=True
)
featurizers['EncodedBonds'] = EncodedBonds

CoulombMatrix = MolMLFeaturizer(
    name='CoulombMatrix',
    fit_atoms=fit_atoms,
)
featurizers['CoulombMatrix'] = CoulombMatrix

Autocorrelation = MolMLFeaturizer(
    name='Autocorrelation',
    fit_atoms=fit_atoms,
)
featurizers['Autocorrelation'] = Autocorrelation

sys.modules['features'] = featurizers
