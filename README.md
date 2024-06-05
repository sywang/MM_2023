# MM_2023

configuration in config.yaml
you can edit the file to run with diffrent config, most scripts get the conf file as an argument
so it is possible to save current configurations

all scripts will create intermediate .h5ad files to /outputs dir
you may need to create one at first run

raw data pipeline until full annotation
1. load_sc_data_to_anndata
2. pp_adata
3. train_scvi_model
4. train_scvi_model_on_sub_population
5. infer annotation (also do clustering)

annoation of malignant cells are not yet automated, and is done after running the annotation inferance

most analysis is done in notebooks
using .h5ad files created by the pipeline
