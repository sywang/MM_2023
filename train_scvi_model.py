import os
from datetime import date
from pathlib import Path
from typing import Optional

import anndata as ad
import scvi
import torch
from omegaconf import OmegaConf

from sc_classification.var_genes import normalize_and_choose_genes, shuang_genes_to_keep


def train_scvi_model(adata_train: ad.AnnData, counts_layer: str = "count",
                     batch_key: Optional[str] = None) -> scvi.model.SCVI:
    scvi.model.SCVI.setup_anndata(
        adata_train,
        layer=counts_layer,
        batch_key=batch_key,
    )
    model = scvi.model.SCVI(adata_train, n_layers=2, dropout_rate=0.2, deeply_inject_covariates=True)

    model.train(batch_size=256, early_stopping=True)
    return model


def generate_model_name(config, extra_description: Optional[str] = None) -> str:
    if config.sc_classification.use_shuang_var_genes != 'None':
        model_name = f"{config.outputs.scvi_model_prefix}_{config.sc_classification.use_shuang_var_genes}_genes"
    else:
        model_name = config.outputs.scvi_model_prefix
    if extra_description is not None:
        model_name += extra_description
    model_name += f"_{date.today().isoformat()}"
    return model_name


def load_pp_adata_to_train(config) -> ad.AnnData:
    adata_path = Path(config.outputs.output_dir, config.outputs.processed_adata_file_name)
    adata = ad.read_h5ad(adata_path)

    adata.layers["counts"] = adata.X.copy()  # preserve counts needed for normalize_and_choose_genes
    genes_to_keep = shuang_genes_to_keep(genes_names=adata.var_names,
                                         flavor=config.sc_classification.use_shuang_var_genes)
    loaded_adata = normalize_and_choose_genes(adata, config, genes_to_keep=genes_to_keep)
    return loaded_adata


if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")

    conf = OmegaConf.load('config.yaml')

    adata_for_training = load_pp_adata_to_train(conf)

    model = train_scvi_model(adata_for_training, counts_layer="count", batch_key="Method")

    model_name = generate_model_name(conf)
    model_path = os.path.join(conf.outputs.output_dir, model_name)
    model.save(model_path, overwrite=True)
