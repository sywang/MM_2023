from enum import Enum

import config
from data_loading.anndata_loaders import FromPlatesDataLoader


class PlatesLoaderDescription(Enum):
    MM_MARS_DATASET = "mm_mars_dataset"
    MM_FULL_DATASET = "mm_full_dataset"  # not implemented


class PlatesDataLoaderFactory:
    @staticmethod
    def create_batch_dataloader(dataloader_description: PlatesLoaderDescription) -> FromPlatesDataLoader:
        if dataloader_description == PlatesLoaderDescription.MM_MARS_DATASET:
            return FromPlatesDataLoader(sc_data_dir=config.MM_MARS_SEQUENCING_DATA_DIR,
                                        plates_data_path=config.MM_BATCH_META_DATA_PATH,
                                        plates_data_transform_functions=[])
        else:
            raise NotImplementedError("need to implement the specfic dataloader")
