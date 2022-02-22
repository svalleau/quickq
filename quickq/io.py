"""MolML transcription tool and save to file functions."""
from typing import Type
import bidict
import os

import numpy as np
import pandas as pd
import ase
import molml.utils

# patch elements which molml does not have all of
molml.utils.ELE_TO_NUM = bidict.bidict(ase.data.atomic_numbers)

Atoms = Type[ase.Atoms]

def atoms_to_molmllist(atoms: Atoms, bonds: bool = False):
    """Constructs a molml list of lists from an atoms object.
    
    By default, will consider Atomic numbers and atom distances, but can be
    directed to also compute bonds. 

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms to compute on.
    bonds : bool, default False
        Whether to compute bonds for the atoms
    
    Returns
    -------
    list of list, used by molml
    """
    positions = atoms.positions
    atomic_numbers = atoms.get_atomic_numbers()
    if bonds:
        raise NotImplementedError(
            'Calculation of molml bond dictionary not implemented yet.'
        )
    else:
        return [atomic_numbers, positions]
    
def save_Q_mols(files_dir, ids, y_hat):
    """Save structure predictions back to file in the temperature table.
    
    Parameters
    ----------
    files_dir : str
        directory containing data files.
    ids : iterable
        Contains ids for each example. Format is specific
        must be of the form XXXXX_Y, where XXXXX is a unique identifier
        for the structure, and Y is an ascending number corresponding to the
        position of a particular example in its' structure's csv file
    y_hat : iterable of float
        Predictions. Same length as ids.
    """
    if not files_dir.endswith('/'):
        files_dir += '/'
    # first determine unique ids
    split_ids = np.array([id_.split('_') for id_ in ids])
    structure_ids, temperature_ids = (split_ids.T)
    output_df = pd.DataFrame(
        {
            'structure_id': structure_ids.reshape(-1),
            'temperature_id': temperature_ids.reshape(-1),
            'log_qpart_predicted': y_hat.reshape(-1)
        }
    )
    output_df['temperature_id'] = output_df['temperature_id'].astype(int)
    # group by structure
    structure_groups = output_df.groupby('structure_id')
    for group_id, structure_df in structure_groups:
        structure_df.sort_values('temperature_id', inplace=True)
        current_df = pd.read_csv(files_dir+group_id+'.csv', index_col=0)
        current_df['log_qpart_predicted'] = structure_df['log_qpart_predicted'].values
        current_df.to_csv(files_dir+group_id+'.csv')
    return

def save_Q_rxns(files_dir, ids, y_hat):
    """Save TS predictions back to file in the temperature table.
    
    Must have temperature as the first column in the original csvs.
    
    Parameters
    ----------
    files_dir : str
        directory containing data files.
    ids : iterable
        Contains ids for each example. Format is specific
        must be of the form XXXXX_Y, where XXXXX is a unique identifier
        for the reaction, and Y is an ascending number corresponding to the
        position of a particular example in its' temperatures
    y_hat : iterable of float
        Predictions. Same length as ids.
    """
    if not files_dir.endswith('/'):
        files_dir += '/'
    split_ids = np.array([id_.split('_') for id_ in ids])
    rxn_ids, temperature_ids = (split_ids.T)
    output_df = pd.DataFrame(
        {
            'rxn_id': rxn_ids.reshape(-1),
            'temperature_id': temperature_ids.reshape(-1),
            'log_qpart_predicted': y_hat.reshape(-1)
        }
    )
    output_df['temperature_id'] = output_df['temperature_id'].astype(int)
    # group by rxn
    rxn_groups = output_df.groupby('rxn_id')
    for rxn_id, rxn_df in rxn_groups:
        rxn_df.sort_values('temperature_id', inplace=True)
        # there may or may not be a ts file already
        if os.path.exists(files_dir+'rxn'+rxn_id+f'/ts{rxn_id}.csv'):
            current_df = pd.read_csv(files_dir+'rxn'+rxn_id+f'/ts{rxn_id}.csv', index_col=0)
        else:
            new_df = pd.read_csv(files_dir+'rxn'+rxn_id+f'/r{rxn_id}.csv', index_col=0)
            new_df = new_df[[new_df.columns[0]]]
            current_df = new_df
        current_df['log_qpart_predicted'] = rxn_df['log_qpart_predicted'].values
        current_df.to_csv(files_dir+'rxn'+rxn_id+f'/ts{rxn_id}.csv')
    return
    
    