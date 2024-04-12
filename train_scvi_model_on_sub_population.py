import os
from typing import List

import torch
import anndata as ad
from omegaconf import OmegaConf

from train_scvi_model import generate_model_name, train_scvi_model, load_pp_adata_to_train


def _get_adata_only_of_populations(adata: ad.AnnData, config, wanted_population_names: List[str]) -> ad.AnnData:
    assert config.annotation.cell_type_columns in adata.obs.columns
    cells_mask = adata.obs[config.annotation.cell_type_columns].apply(lambda x: x in wanted_population_names)
    assert sum(cells_mask) > 0, "no cells found of the given populations"
    return adata[cells_mask]


if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")

    conf = OmegaConf.load('config.yaml')

    adata_for_training = load_pp_adata_to_train(conf)

    population_names = ["Malignant", "Healthy"]
    adata_for_training = _get_adata_only_of_populations(adata_for_training, conf, population_names)

    model = train_scvi_model(adata_for_training, counts_layer="count", batch_key="Method")

    model_name = generate_model_name(conf, extra_description=f"_only_{'_'.join(population_names)}")
    model_path = os.path.join(conf.outputs.output_dir, model_name)
    model.save(model_path, overwrite=True)
