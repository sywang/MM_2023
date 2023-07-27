import argparse
import logging
import os
import sys
from pathlib import Path

import anndata as ad
from omegaconf import OmegaConf, DictConfig

sys.path.append(os.getcwd())

from data_loading.data_loader_factory import create_dataloader_from_config
from logging_utils import set_file_logger


def load_sc_data_to_anndata(config: DictConfig) -> ad.AnnData:
    adata_loader = create_dataloader_from_config(config)
    adata = adata_loader.load_data_to_anndata()
    output_file_name = Path(config.outputs.output_dir, config.outputs.loaded_adata_file_name)
    if output_file_name is not None:
        adata.write(output_file_name)
        logging.info(f"saving AnnData to file - {output_file_name}")
    return adata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='AnnData Loading',
        description='loads scRNA data to AnnData and save it to h5ad file')

    parser.add_argument('--config', help='a path to an valid config file', default='config.yaml')

    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    logging_file_path = Path(conf.outputs.output_dir, conf.outputs.logging_file_name)
    set_file_logger(logging_file_path)

    load_sc_data_to_anndata(config=conf)
