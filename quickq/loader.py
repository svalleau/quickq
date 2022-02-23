"""Loader of raw data into deepchem dataset after featurization.

Qest loader creates datasets of featurized molecules.
QesTS loader creates datasets of featurized reactions.
Double loader creates dataset of featurized reactants and products,
makes a prediction with Qest, and uses this to produce a dataset of
featurized reactions.
"""
from typing import Union, Type, Iterable, List, Iterator
from pathlib import Path
import logging
import time
import os

import ase
import deepchem.data
import numpy
import pandas as pd

import quickq.structure
import quickq.featurizers

logger = logging.getLogger(__name__)

class QestLoader:
    """Loads molecules and their Q values from raw data files.
    
    Data must be stored in a folder alone as:
    -XXX.extxyz
    -XXX.csv
    For each molecule XXX. csv must contain column "T".
    Data is featurized and saved as a deepchem dataset.
    
    Parameters
    ----------
    featurizer : quickq.featurizers.MolFeaturizer
        Featurizer to apply to each molecule
    """
    def __init__(
        self,
        featurizer: quickq.featurizers.MolFeaturizer = None
    ):
        self.featurizer = featurizer
        return
    
    def _get_shards(
        self,
        files_dir: str,
        shard_size: int,
        num_shards: int
    ) -> Iterator:
        """Shardize the files_dir directory and return a generator for the shards.
        
        Parameters
        ----------
        files_dir : str
            directory containing the data. See class docs for details.
        shard_size : int
            Number of structures to load per shard
        num_shards : int
            number of shards from total to load.
        
        Returns
        -------
        generator of shards
        """
        # iterate through shards
        shard_num = 1
        # get a big list of the reactions
        data_paths = [files_dir+str(path) for path in os.listdir(files_dir) if path.endswith('.extxyz')]
        logger.info(f'Total shards: {int(len(data_paths)/shard_size)}')
        for shard_indexes in range(0, len(data_paths), shard_size):
            # if we haven't reached out shard limit, open the shard
            if num_shards is None or shard_num <= num_shards:
                shardpaths = data_paths[shard_indexes:shard_indexes+shard_size]
                logger.info(f'Loading shard {shard_num}')
                shard_num += 1
                yield self._open_shard(shardpaths)
            else:
                break
            
    def _open_shard(self, shardpaths: List[str]):
        """Open a single list of files into structures.
        
        Parameters
        ----------
        shardpaths : list of str
            The paths to structures in this shard
        
        Returns
        -------
        structures : list of Structure objects
        ind : list of structure indexes
        """
        structures = []
        ind = []
        for path in shardpaths:
            no_extension = path[:-7]
            idx = path.split('/')[-1][:-7]
            struc = quickq.structure.Structure.load_properties(
                path, csv_filename = no_extension+'.csv'
            )
            structures.append(struc)
            ind.append(idx)
        return structures, ind
    
    def load_data(self,
                  files_dir: str,
                  shard_size: int = 500,
                  num_shards: int = None
                 ):
        """Load the data into pandas dataframes.
        
        Parameters
        ----------
        files_dir : str
            directory containing the data. See class docs for details.
        shard_size : int
            Number of structures to load per shard
        num_shards : int,
            number of shards from total to load.
        
        Returns
        -------
        generator of dataframes
        """
        logger.info("Loading raw samples now.")
        logger.info("shard_size: %s" % str(shard_size))
        if not files_dir.endswith('/'):
            files_dir += '/'
        
        def shard_generator():
            for shard_num, shard in enumerate(self._get_shards(files_dir, shard_size, num_shards)):
                time1 = time.time()
                structures, ind = shard
                # featurize the molprops
                if self.featurizer is not None:
                    feats = self.featurizer.featurize(structures)
                    
                dfs = []
                for i, struc in enumerate(structures):
                    # we need to expand each mol on the temperature and
                    # Q vector
                    df = pd.DataFrame({'T':list(struc.T.flatten())})
                    if struc.log_qpart is not None:
                        df['logQ'] = list(struc.log_qpart.flatten())
                    
                    # try featurization if present
                    if self.featurizer is not None:
                        df[self.featurizer.name] = list(numpy.tile(feats[i], (len(df), 1)))
                        
                    df['ids'] = df.apply(lambda row: ind[i]+'_'+str(int(row.name)), axis=1)
                    dfs.append(df)
                df = pd.concat(dfs)

                time2 = time.time()
                logger.info("TIMING: featurizing shard %d took %0.3f s" %
                            (shard_num, time2 - time1))
                
                yield df
                
        return shard_generator()
    
    def create_dataset(self,
                       files_dir: str,
                       data_dir: str,
                       shard_size: int = 500,
                       num_shards: int = None
                       ) -> deepchem.data.DiskDataset:
        """Featurize raw data into deepchem dataset.
        
        Parameters
        ----------
        files_dir : str
            directory containing the data. See class docs for details.
        data_dir : str
            directory name to store deepchem disk dataset
        shard_size : int
            Number of structures to load per shard
        num_shards : int
            number of shards from total to load.
        """
        def shard_generator():
            for df in self.load_data(files_dir=files_dir, shard_size=shard_size, num_shards=num_shards):
                # add temperature to whatever feature vector was computed
                feats = numpy.vstack(df[self.featurizer.name].values)
                T = df['T'].values.reshape(-1,1)
                X = numpy.append(feats, 1/T, axis=1)
                
                if 'logQ' in df.columns:
                    y = df['logQ'].values.reshape(-1,1)
                else:
                    y= numpy.empty(len(X))
                w = numpy.ones(len(X))
                
                ids = numpy.array(df['ids']).reshape(-1,1)
                yield X, y, w, ids
        return deepchem.data.DiskDataset.create_dataset(shard_generator(), data_dir, ['logQ'])
    
class QesTSLoader:
    """Loads structures from reactions from raw data files.
    
    Data for each reaction must be stored in a folder as:
    rxnXXX/
    -rXXX.extxyz
    -rXXX.csv
    -pXXX.extxyz
    -pXXX.csv
    For each reaction XXX. csvs must contain column temperature "T" as first columns
    and "log_qpart" as the reactants/products logged Q values. T values must match.
    Data is featurized and saved as a deepchem dataset.
    
    Parameters
    ----------
    featurizer : quickq.featurizers.MolFeaturizer
        Featurizer to apply to each molecule
    """
    def __init__(
        self,
        featurizer: quickq.featurizers.MolFeaturizer = None
    ):
        self.featurizer = featurizer
        return

    def _get_shards(
        self,
        files_dir: str,
        shard_size: int,
        num_shards: int
    ) -> Iterator:
        """Shardize the files_dir directory and return a generator for the shards.
        
        Parameters
        ----------
        files_dir : str
            directory containing the data. See class docs for details.
        shard_size : int
            Number of reactions to load per shard
        num_shards : int
            number of shards from total to load.
        
        Returns
        -------
        generator of shards
        """
        # iterate through shards
        shard_num = 1
        # get a big list of the reactions
        rxn_paths = [files_dir+str(path) for path in os.listdir(files_dir)]
        logger.info(f'Total shards: {int(len(rxn_paths)/shard_size)}')
        for shard_indexes in range(0, len(rxn_paths), shard_size):
            # if we haven't reached out shard limit, open the shard
            if num_shards is None or shard_num <= num_shards:
                shardpaths = rxn_paths[shard_indexes:shard_indexes+shard_size]
                logger.info(f'Loading shard {shard_num}')
                shard_num += 1
                yield self._open_shard(shardpaths)
            else:
                break
                
    
    def _open_shard(self, shardpaths: List[str]):
        """Open a single list of reaction directories into structures.
        
        Parameters
        ----------
        shardpaths : list of str
            The paths to reactions in this shard
        
        Returns
        -------
        structures : list of list of Structure objects
        rxns : list of reaction indexes
        """
        rxns = []
        structures = []
        for rxn_path in shardpaths:
            rxn = rxn_path.split('/')[-1][3:]
            # reactant, product, ts
            r = quickq.structure.Structure.load_properties(
                rxn_path+'/r'+rxn+'.extxyz',
                csv_filename = rxn_path+'/r'+rxn+'.csv'
            )
            p = quickq.structure.Structure.load_properties(
                rxn_path+'/p'+rxn+'.extxyz',
                csv_filename = rxn_path+'/p'+rxn+'.csv'
            )
            try:
                ts = quickq.structure.Structure.load_properties(
                    rxn_path+'/ts'+rxn+'.extxyz',
                    csv_filename = rxn_path+'/ts'+rxn+'.csv'
                )
            except:
                ts = None
            structures.append([r, p, ts])
            rxns.append(rxn)
            
            # if we cannot produce a scaffold just continue
        return structures, rxns
    
    def load_data(self,
                  files_dir: str,
                  shard_size: int = 500,
                  num_shards: int = None
                 ):
        """Load the data into pandas dataframes.
        
        Parameters
        ----------
        files_dir : str
            directory containing the data. See class docs for details.
        shard_size : int
            Number of reactions to load per shard
        num_shards : int
            number of shards from total to load.
        
        Returns
        -------
        generator of dataframes
        """
        logger.info("Loading raw samples now.")
        logger.info("shard_size: %s" % str(shard_size))
        
        def shard_generator():
            for shard_num, shard in enumerate(self._get_shards(files_dir, shard_size, num_shards)):
                time1 = time.time()
                structures, rxns = shard
                    
                # loop through each reaction, NOT each molecule
                dfs=[]
                for i, rxn in enumerate(rxns):
                    structure_set = structures[i]
                    # check we haev the expected sizes
                    assert len(structure_set) == 3, 'rxn should have 3 systems'
                    assert len(numpy.unique([len(mp.atoms) for mp in structure_set if mp is not None])) == 1, 'all systems not the same size'
                    # create dataframe of T dependant quantities
                    if structure_set[0].log_qpart is None or structure_set[1].log_qpart is None:
                        raise ValueError('Cannot use QesTS predictor without R and P partition function')
                    df = pd.DataFrame({'T':list(structure_set[0].T.flatten()),
                                       'logQr':list(structure_set[0].log_qpart.flatten()),
                                       'logQp':list(structure_set[1].log_qpart.flatten()),
                                      })
                    if structure_set[2] is not None and structure_set[2].log_qpart is not None:
                        df['logQts'] = list(structure_set[2].log_qpart.flatten())
                    
                    # get the features difference
                    if self.featurizer is not None:
                        rfeats = self.featurizer.featurize(structure_set[0])
                        pfeats = self.featurizer.featurize(structure_set[1])
                        feats = pfeats - rfeats
                        # add it the the df, all rows have the same value because these features on not
                        # temperature dependant
                        df[self.featurizer.name] = list(numpy.tile(feats, (len(df), 1)))
                    # set a row of ids
                    df['ids'] = df.apply(lambda row: rxns[i]+'_'+str(int(row.name)), axis=1)
                    dfs.append(df)
               
                # combine all reactions in this shard
                df = pd.concat(dfs)

                time2 = time.time()
                logger.info("TIMING: featurizing shard %d took %0.3f s" %
                            (shard_num, time2 - time1))
                
                yield df
                
        return shard_generator()
    
    def create_dataset(self,
                       files_dir: str,
                       data_dir: str,
                       shard_size: int = 500,
                       num_shards: int = None
                       ) -> deepchem.data.DiskDataset:
        """Featurize raw data into deepchem dataset.
        
        Parameters
        ----------
        files_dir : str
            directory containing the data. See class docs for details.
        data_dir : str
            directory name to store deepchem disk dataset
        shard_size : int
            Number of reactions to load per shard
        num_shards : int
            number of shards from total to load.
        """
        if not files_dir.endswith('/'):
            files_dir +='/'
        def shard_generator():
            for df in self.load_data(files_dir=files_dir, shard_size=shard_size, num_shards=num_shards):
                # add temperature to whatever feature vector was computed
                feats = numpy.vstack(df[self.featurizer.name].values)
                qr = df['logQr'].values.reshape(-1,1)
                qp = df['logQp'].values.reshape(-1,1)
                Tinv = 1/df['T'].values.reshape(-1,1)
                X = numpy.concatenate([feats, qr, qp, Tinv], axis=1)
                
                if 'logQts' in df.columns:
                    y = df['logQts'].values.reshape(-1,1)
                else:
                    y= numpy.empty(len(X))
                w = numpy.ones(len(X))
                
                ids = numpy.array(df['ids']).reshape(-1,1)
                yield X, y, w, ids
        return deepchem.data.DiskDataset.create_dataset(shard_generator(), data_dir, ['logQts'])
    
    
class DoubleLoader:
    """Loads structures from reactions from raw data files.
    
    Data for each reaction must be stored in a folder as:
    rxnXXX/
    -rXXX.extxyz
    -rXXX.csv
    -pXXX.extxyz
    -pXXX.csv
    For each reaction XXX. csvs must contain column temperature "T" as first columns.
    T values must match.
    Data is featurized and saved as a deepchem dataset.
    
    Parameters
    ----------
    featurizer : quickq.featurizers.MolFeaturizer
        Featurizer to apply to each molecule
    """
    def __init__(
        self,
        featurizer: quickq.featurizers.MolFeaturizer = None
    ):
        self.featurizer = featurizer
        import quickq.pipeline
        return

    def _get_shards(
        self,
        files_dir: str,
        shard_size: int,
        num_shards: int
    ) -> Iterator:
        """Shardize the files_dir directory and return a generator for the shards.
        
        Parameters
        ----------
        files_dir : str
            directory containing the data. See class docs for details.
        shard_size : int
            Number of reactions to load per shard
        num_shards : int
            number of shards from total to load.
        
        Returns
        -------
        generator of shards
        """
        # iterate through shards
        shard_num = 1
        # get a big list of the reactions
        rxn_paths = [files_dir+str(path) for path in os.listdir(files_dir)]
        logger.info(f'Total shards: {int(len(rxn_paths)/shard_size)}')
        for shard_indexes in range(0, len(rxn_paths), shard_size):
            # if we haven't reached out shard limit, open the shard
            if num_shards is None or shard_num <= num_shards:
                shardpaths = rxn_paths[shard_indexes:shard_indexes+shard_size]
                logger.info(f'Loading shard {shard_num}')
                shard_num += 1
                yield self._open_shard(shardpaths)
            else:
                break
                
    
    def _open_shard(self, shardpaths: List[str]):
        """Open a single list of reaction directories into structures.
        
        Parameters
        ----------
        shardpaths : list of str
            The paths to reactions in this shard
        
        Returns
        -------
        structures : list of list of Structure objects
        rxns : list of reaction indexes
        """
        rxns = []
        structures = []
        for rxn_path in shardpaths:
            rxn = rxn_path.split('/')[-1][3:]
            # reactant, product, ts
            r = quickq.structure.Structure.load_properties(
                rxn_path+'/r'+rxn+'.extxyz',
                csv_filename = rxn_path+'/r'+rxn+'.csv'
            )
            p = quickq.structure.Structure.load_properties(
                rxn_path+'/p'+rxn+'.extxyz',
                csv_filename = rxn_path+'/p'+rxn+'.csv'
            )
            try:
                ts = quickq.structure.Structure.load_properties(
                    rxn_path+'/ts'+rxn+'.extxyz',
                    csv_filename = rxn_path+'/ts'+rxn+'.csv'
                )
            except:
                ts = None
            structures.append([r, p, ts])
            rxns.append(rxn)
            
            # if we cannot produce a scaffold just continue
        return structures, rxns
    
    def load_data(self,
                  files_dir: str,
                  shard_size: int = 500,
                  num_shards: int = None
                 ):
        """Load the reactant and product data, make predictions, then give dataframes
        
        Parameters
        ----------
        files_dir : str
            directory containing the data. See class docs for details.
        shard_size : int
            Number of reactions to load per shard
        num_shards : int
            number of shards from total to load.
        
        Returns
        -------
        generator of dataframes
        """
        logger.info("Loading raw samples now.")
        logger.info("shard_size: %s" % str(shard_size))
        
        def shard_generator():
            for shard_num, shard in enumerate(self._get_shards(files_dir, shard_size, num_shards)):
                time1 = time.time()
                structures, rxns = shard
                    
                # loop through each reaction, NOT each molecule
                dfs=[]
                for i, rxn in enumerate(rxns):
                    structure_set = structures[i]
                    # check we haev the expected sizes
                    assert len(structure_set) == 3, 'rxn should have 3 systems'
                    assert len(numpy.unique([len(mp.atoms) for mp in structure_set if mp is not None])) == 1, 'all systems not the same size'
                    # create dataframe of T dependant quantities
                    df = pd.DataFrame({'T':list(structure_set[0].T.flatten()),
                                      })
                    if structure_set[2] is not None and structure_set[2].log_qpart is not None:
                        df['logQts'] = list(structure_set[2].log_qpart.flatten())
                    
                    # get the features difference
                    if self.featurizer is not None:
                        rfeats = self.featurizer.featurize(structure_set[0])
                        pfeats = self.featurizer.featurize(structure_set[1])
                        feats = pfeats - rfeats
                        # add it the the df, all rows have the same value because these features on not
                        # temperature dependant
                        df[self.featurizer.name] = list(numpy.tile(feats, (len(df), 1)))
                        
                    # predict the Qs with qest
                    rfeats = numpy.concatenate([numpy.tile(rfeats, (len(df), 1)), (1/df['T'].values).reshape(-1,1)], axis=1)
                    r_dataset = deepchem.data.NumpyDataset(rfeats)
                    logQr = quickq.pipeline.predict_qest(r_dataset)
                    pfeats = numpy.concatenate([numpy.tile(pfeats, (len(df), 1)), (1/df['T'].values).reshape(-1,1)], axis=1)
                    p_dataset = deepchem.data.NumpyDataset(pfeats)
                    logQp = quickq.pipeline.predict_qest(p_dataset)
                    
                    df['logQr'] = logQr
                    df['logQp'] = logQp
                    
                    # set a row of ids
                    df['ids'] = df.apply(lambda row: rxns[i]+'_'+str(int(row.name)), axis=1)
                    dfs.append(df)
               
                # combine all reactions in this shard
                df = pd.concat(dfs)

                time2 = time.time()
                logger.info("TIMING: featurizing shard %d took %0.3f s" %
                            (shard_num, time2 - time1))
                
                yield df
                
        return shard_generator()
    
    def create_dataset(self,
                       files_dir: str,
                       data_dir: str,
                       shard_size: int = 500,
                       num_shards: int = None
                       ) -> deepchem.data.DiskDataset:
        """Featurize raw data into deepchem dataset.
        
        Parameters
        ----------
        files_dir : str
            directory containing the data. See class docs for details.
        data_dir : str
            directory name to store deepchem disk dataset
        shard_size : int
            Number of reactions to load per shard
        num_shards : int
            number of shards from total to load.
        """
        if not files_dir.endswith('/'):
            files_dir += '/'
        def shard_generator():
            for df in self.load_data(files_dir=files_dir, shard_size=shard_size, num_shards=num_shards):
                # add temperature to whatever feature vector was computed
                feats = numpy.vstack(df[self.featurizer.name].values)
                qr = df['logQr'].values.reshape(-1,1)
                qp = df['logQp'].values.reshape(-1,1)
                Tinv = 1/df['T'].values.reshape(-1,1)
                X = numpy.concatenate([feats, qr, qp, Tinv], axis=1)
                
                if 'logQts' in df.columns:
                    y = df['logQts'].values.reshape(-1,1)
                else:
                    y= numpy.empty(len(X))
                w = numpy.ones(len(X))
                
                ids = numpy.array(df['ids']).reshape(-1,1)
                yield X, y, w, ids
        return deepchem.data.DiskDataset.create_dataset(shard_generator(), data_dir, ['logQts'])
