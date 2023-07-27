from functools import partial
from pathlib import Path
from typing import List, Optional

import anndata as ad
import pandas as pd

_supported_suffixes_to_reader = {
    ".csv": pd.read_csv,
    ".xlsx": pd.read_excel,
    ".txt": partial(pd.read_csv, delimiter="\t")
}


def load_dataframe_from_file(plates_data_path: Path) -> pd.DataFrame:
    path_suffix = plates_data_path.suffix
    if path_suffix in _supported_suffixes_to_reader:
        plates_data_df = _supported_suffixes_to_reader[path_suffix](plates_data_path)
        return plates_data_df
    else:
        raise ValueError(f"Not supperted meta_data_file_type:"
                         f" got {path_suffix}, supported are {list(_supported_suffixes_to_reader)}")


def merge_labels_to_adata(adata: ad.AnnData, labels_df: pd.DataFrame, col_in_adata_to_merge_by: str,
                          cols_in_labels_df_to_merge_by: str, cols_to_validate_not_empty: List[str],
                          labels_col_names: Optional[List[str]] = None, merge_suffixes=("_x", "_y")):
    if labels_col_names is None:
        labels_col_names = labels_df.columns
    if cols_in_labels_df_to_merge_by == "index":
        labels_df = labels_df.reset_index()
    if col_in_adata_to_merge_by == "index":
        adata.obs.index = adata.obs.index.map(str)
    else:
        adata.obs = adata.obs.astype({col_in_adata_to_merge_by: "str"})
    adata.obs = pd.merge(
        left=adata.obs.reset_index(),
        right=labels_df[labels_col_names], how="left", validate="m:1", suffixes=merge_suffixes,
        left_on=col_in_adata_to_merge_by, right_on=cols_in_labels_df_to_merge_by).set_index("index")
    adata = adata[~ adata.obs[cols_to_validate_not_empty].isna().any(axis=1), :]
