import logging
import os
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional

import anndata as ad
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from tqdm.contrib.concurrent import process_map

from data_loading.meta_data_columns_names import BATCH_ID
from data_loading.meta_data_loader import load_metadata_from_file


class AnnDataLoader(ABC):

    @abstractmethod
    def load_data_to_anndata(self) -> ad.AnnData:
        pass


DEBUG_MODE = False
DEBUG_N_BATCHES = 10
BAD_COLUMNS_NAMES = []


class BatchDataLoader(AnnDataLoader):
    def __init__(self, experiments_data_dir: Path, meta_data_path: Path,
                 metadata_transform_functions: Optional[List[Callable[[pd.DataFrame], pd.DataFrame]]] = None):
        self.experiments_data_dir = experiments_data_dir
        self.meta_data_path = meta_data_path
        self.metadata_transform_functions = metadata_transform_functions if metadata_transform_functions is not None else []

    def _get_single_batch(self, row_tpl, col_names):
        row = row_tpl[1]
        logging.info(f"Reading , batch id - {row[BATCH_ID]}")
        cur_data = sc.read_text(Path(self.experiments_data_dir, row[BATCH_ID] + ".txt"))
        cur_data = cur_data.T
        logging.info("inserting metadata to the anndata")
        for col_name in col_names:
            cur_data.obs[col_name] = row[col_name]
        logging.info(f"converting data from batch id - {row[BATCH_ID]} to sparse matrix")
        cur_data.X = csr_matrix(cur_data.X)
        return cur_data

    def load_data_to_anndata(self) -> ad.AnnData:
        # Read annotation file
        metadata_df = load_metadata_from_file(self.meta_data_path)
        metadata_df = metadata_df.dropna(axis=0, subset=[BATCH_ID])
        if DEBUG_MODE:
            self.metadata_transform_functions.append(lambda df: df.head(DEBUG_N_BATCHES))
        for transform_func in self.metadata_transform_functions:
            logging.info(f"applying the function: {transform_func.__name__} on the meta data")
            metadata_df = transform_func(metadata_df)

        # Read all plates into anndata and merge them
        col_names = metadata_df.columns
        if DEBUG_MODE:
            adatas = [self._get_single_batch(batch_tpl, col_names) for batch_tpl in metadata_df.iterrows()]
        else:
            adatas = process_map(partial(self._get_single_batch, col_names=col_names),
                                 list(metadata_df.iterrows()), max_workers=os.cpu_count() // 2,
                                 desc="loading relevant batches",
                                 unit="batch")
        logging.info("merging to single adata")
        adata = ad.concat(adatas, merge="same")

        adata.obs = adata.obs.astype(str)

        if len(BAD_COLUMNS_NAMES) > 0:
            logging.warning(f"dropping - {BAD_COLUMNS_NAMES} columns, some bug with that columns")
            adata.obs.drop(BAD_COLUMNS_NAMES, axis='columns', inplace=True)

        return adata
