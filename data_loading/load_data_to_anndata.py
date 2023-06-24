import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import anndata as ad

sys.path.append(os.getcwd())

from data_loading.batch_data_loader_factory import PlatesDataLoaderFactory, PlatesLoaderDescription


def load_data_to_anndata(file_name: Optional[Path],
                         loader_description: PlatesLoaderDescription = PlatesLoaderDescription.MM_MARS_DATASET) \
        -> ad.AnnData:
    adata_loader = PlatesDataLoaderFactory().create_batch_dataloader(loader_description)
    adata = adata_loader.load_data_to_anndata()
    if file_name is not None:
        adata.write(file_name)
        logging.info(f"saving AnnData to file - {file_name}")
    return adata


if __name__ == '__main__':
    logging.basicConfig(filename='lust_run.log', level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog='AnnData Loading',
        description='loads scRNA data to AnnData and save it to h5ad file')

    parser.add_argument('file_name', help='path to write the h5ad file to')

    args = parser.parse_args()

    load_data_to_anndata(file_name=args.file_name)
