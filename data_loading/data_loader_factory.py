from pathlib import Path

from omegaconf import DictConfig

from data_loading.anndata_loaders import FromPlatesDataLoader, AnnDataLoader


def create_dataloader_from_config(config: DictConfig) -> AnnDataLoader:
    return FromPlatesDataLoader(sc_data_dir=Path(config.data_loading.sc_sequencing.sc_sequencing_data_dir),
                                plates_data_path=Path(config.data_loading.plates.plates_data_path),
                                plate_id_col_name=config.data_loading.plates.plate_id_column_name,
                                plates_data_transform_functions=[])
