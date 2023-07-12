import argparse
import logging
from pathlib import Path

import anndata as ad
from omegaconf import OmegaConf, DictConfig

from load_data_to_anndata import load_data_to_anndata


def pre_process(config: DictConfig):
    adata_path_from_config = Path(config.data_loading.loaded_adata_dir, config.data_loading.loaded_adata_file_name)
    if adata_path_from_config.exists():
        adata = ad.read_h5ad(adata_path_from_config)
    else:
        adata = load_data_to_anndata(config)

    load_annotations(adata)
    drop_bad_genes(adata)
    drop_bad_cells(adata)

    processed_adata_path = Path(config.pre_processing.processed_adata_dir,
                                config.pre_processing.processed_adata_file_name)
    if processed_adata_path is not None:
        adata.write(processed_adata_path)
        logging.info(f"saving processed AnnData to file - {processed_adata_path}")
    return adata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='AnnData Loading',
        description='loads scRNA data to AnnData and save it to h5ad file')

    parser.add_argument('--config', help='a path to an valid config file', default='config.yaml')

    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    pre_process(config=conf)
