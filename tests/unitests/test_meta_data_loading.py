from pathlib import Path

from data_loading.utils import load_dataframe_from_file


def test_load_metadata_from_file_txt():
    test_file_path = Path("tests/data/test_meta_data.txt")

    meta_data_df = load_dataframe_from_file(test_file_path)

    assert len(meta_data_df) >= 0
    assert 'Amp.Batch.ID' in meta_data_df.columns
