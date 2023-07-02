from pathlib import Path

from data_loading.meta_data_columns_names import PLATE_ID
from data_loading.plates_data_loader import load_plates_data_from_file


def test_load_metadata_from_file_txt():
    test_file_path = Path("tests/data/test_meta_data.txt")

    meta_data_df = load_plates_data_from_file(test_file_path)

    assert len(meta_data_df) >= 0
    assert PLATE_ID in meta_data_df.columns
