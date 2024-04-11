import os
from datetime import date
from pathlib import Path

import anndata as ad
import pandas as pd
import scvi
import torch
from omegaconf import OmegaConf

from data_loading.utils import merge_labels_to_adata
from sc_classification.var_genes import normalize_and_choose_genes, shuang_genes_to_keep

torch.set_float32_matmul_precision("high")

conf = OmegaConf.load('config.yaml')

adata_path = Path(conf.outputs.output_dir, conf.outputs.processed_adata_file_name)
adata = ad.read_h5ad(adata_path)

adata.layers["counts"] = adata.X.copy()  # preserve counts needed for normalize_and_choose_genes

genes_to_keep = shuang_genes_to_keep(genes_names=adata.var_names, flavor=conf.sc_classification.use_shuang_var_genes)
adata_for_clustering = normalize_and_choose_genes(adata, conf, genes_to_keep=genes_to_keep)

# train only on labeled PC
population_name = "PC"
super_pop = "super_Population"
cells_dir = "/home/labs/amit/noamsh/data/mm_2023/cells"
all_annotation_path = Path(cells_dir, "cells_snnotation_20231110.csv")
all_labels = pd.read_csv(all_annotation_path)
adata_for_clustering = merge_labels_to_adata(adata_for_clustering, all_labels,
                                             col_in_adata_to_merge_by="index",
                                             cols_in_labels_df_to_merge_by="cID",
                                             cols_to_validate_not_empty=[super_pop],
                                             labels_col_names_to_merge=[super_pop])
adata_for_clustering = adata_for_clustering[adata_for_clustering.obs[super_pop] == population_name].copy()
# adata_for_clustering.obs = adata_for_clustering.obs.rename(columns={"Hospital.Code": "Hospital-Code"})

scvi.model.SCVI.setup_anndata(
    adata_for_clustering,
    layer="counts",
    batch_key="Method",
    # categorical_covariate_keys=["Hospital-Code"]
)
model = scvi.model.SCVI(adata_for_clustering, n_layers=2, dropout_rate=0.2, deeply_inject_covariates=True)

model.train(batch_size=256, early_stopping=True)

if conf.sc_classification.use_shuang_var_genes != 'None':
    model_name = f"{conf.outputs.scvi_model_name}_{conf.sc_classification.use_shuang_var_genes}_genes"
else:
    model_name = conf.outputs.scvi_model_name

model_name += f"_only_{population_name}"
model_name += f"_{date.today().isoformat()}"

model_path = os.path.join(conf.outputs.output_dir, model_name)
model.save(model_path, overwrite=True)

