from typing import List, Optional

import pandas as pd


def merge_transcriptom_data_to_raw_hospital(transcriptome_dataset: pd.DataFrame,
                                            raw_hospital_dataset: pd.DataFrame,
                                            filter_transcriptome_dataset_by_col: Optional[str] = "Transcriptom",
                                            transcriptome_dataset_patient_id_col_name: Optional[str] = "PID") -> pd.DataFrame:
    dataset = transcriptome_dataset
    if filter_transcriptome_dataset_by_col is not None:
        dataset = transcriptome_dataset[transcriptome_dataset[filter_transcriptome_dataset_by_col].fillna(False)]
    if "Unnamed: 0" in dataset.columns:
        dataset = dataset.drop(columns=["Unnamed: 0"])
    if transcriptome_dataset_patient_id_col_name is not None:
        dataset = dataset.set_index("PID")

    ## add post treatment columns
    post_treatment_cols = [col for col in raw_hospital_dataset.columns if ".2" in col]
    # add fish_columns
    fish_cols = [col for col in raw_hospital_dataset.columns if
                 "t(" in col or "del(" in col or col in ['1q21+', 'IGH rearrangement',
                                                         'Cytogenetics Risk (0=standard risk, 1=single hit, 2=2+ hits)']]
    post_treatment_data = raw_hospital_dataset[raw_hospital_dataset["Time"] != "Post"][ #TODO not good to throw "Post", nned maybe
        ["Code"] + post_treatment_cols + fish_cols].set_index("Code")
    dataset = dataset.merge(post_treatment_data, how="left", left_index=True, right_index=True, validate="one_to_one")

    ## add data to TAL_3 patients
    TAL3_patients = [code for code in dataset.index if "P" == code[0]]
    dataset.loc[TAL3_patients, "Stage"] = 3
    dataset.loc[TAL3_patients, "Lenalidomide"] = 2

    return dataset


def generate_refracrotines_dataset(dataset: pd.DataFrame, treatment: str, non_ref_policy: str, feats: List[str]):
    pre_biopsy_ref_mask = dataset[treatment] == 2
    post_biopsy_ref_mask = dataset[f"{treatment}.2" if treatment != "DARA" else "Daratumumab.2"] == 2
    ref_mask = pre_biopsy_ref_mask | post_biopsy_ref_mask

    NDMM_STAGE_LIST = [1, 2]
    if non_ref_policy == "NDMM":
        newly_diagnosed = dataset["Stage"].apply(lambda x: x in NDMM_STAGE_LIST)
        non_ref_mask = newly_diagnosed
    elif non_ref_policy == "NDMM-POST_TREATMENT_REF":
        newly_diagnosed = dataset["Stage"].apply(lambda x: x in NDMM_STAGE_LIST)
        non_ref_mask = (newly_diagnosed) | (dataset[f"{treatment}.2"] == 4)
    elif non_ref_policy == "NON_EXPOSED":
        non_ref_mask = dataset[treatment].isna()
    else:
        raise NotImplementedError

    X = pd.concat([dataset[ref_mask][feats], dataset[non_ref_mask][feats]], axis=0)
    y = pd.concat([ref_mask[ref_mask].astype(int), non_ref_mask[non_ref_mask].astype(int) - 1])

    return X, y
