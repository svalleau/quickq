"""Defines the featurizer used to make predictions

MolML requires the systems themselves to fit the featurizer. This is a problem
when we featurize individual chunks at a time. We must first retrieve the fit
atoms.
"""

import ase.io
from quickq.featurizers MolMLFeaturizer
import sys
import os
filepath = os.path.dirname(os.path.realpath(__file__))
fit_atoms_dir = os.path.join(filepath, 'fit_atoms')

# load the fit atoms for molml featurizers
fit_atoms = [ase.io.read(fit_atoms_dir+'/'+system) for system in os.listdir(fit_atoms_dir) if system.endswith('xyz')]

# define each featurizer
featurizers = {}
EncodedBonds = MolMLFeaturizer(
    name='EncodedBond',
    fit_atoms=fit_atoms,
    add_unknown=True
)
featurizers['EncodedBonds'] = EncodedBonds

sys.modules['features'] = featurizers
