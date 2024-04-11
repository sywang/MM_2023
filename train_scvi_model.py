import os
from datetime import date
from pathlib import Path

import anndata as ad
import scvi
import torch
from omegaconf import OmegaConf

from sc_classification.var_genes import normalize_and_choose_genes, shuang_genes_to_keep

torch.set_float32_matmul_precision("high")

conf = OmegaConf.load('config.yaml')

adata_path = Path(conf.outputs.output_dir, conf.outputs.processed_adata_file_name)
adata = ad.read_h5ad(adata_path)

adata.layers["counts"] = adata.X.copy()  # preserve counts needed for normalize_and_choose_genes

genes_to_keep = shuang_genes_to_keep(genes_names=adata.var_names, flavor=conf.sc_classification.use_shuang_var_genes)
adata_for_clustering = normalize_and_choose_genes(adata, conf, genes_to_keep=genes_to_keep)

scvi.model.SCVI.setup_anndata(
    adata_for_clustering,
    layer="counts",
    batch_key="Method",
)
model = scvi.model.SCVI(adata_for_clustering, n_layers=2, dropout_rate=0.2, deeply_inject_covariates=True)

model.train(batch_size=256, early_stopping=True)

if conf.sc_classification.use_shuang_var_genes != 'None':
    model_name = f"{conf.outputs.scvi_model_name}_{conf.sc_classification.use_shuang_var_genes}_genes"
else:
    model_name = conf.outputs.scvi_model_name

model_name += f"_{date.today().isoformat()}"

model_path = os.path.join(conf.outputs.output_dir, model_name)
model.save(model_path, overwrite=True)
