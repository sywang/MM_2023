import logging

import anndata as ad
import scanpy as sc


def pp_drop_cells(adata, min_num_genes_per_cell: int, min_num_counts_per_cell: int,
                  mitochondrial_prefix: str, max_pct_mt_genes_pre_cell: float):
    logging.info("drop bad cells")
    sc.pp.filter_cells(adata, min_genes=min_num_genes_per_cell)
    sc.pp.filter_cells(adata, min_counts=min_num_counts_per_cell)

    adata.var['mt'] = adata.var_names.str.startswith(mitochondrial_prefix)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < max_pct_mt_genes_pre_cell, :]


def pp_drop_genes(adata: ad, min_num_cells_per_gene):
    logging.info("drop bad genes")
    sc.pp.filter_genes(adata, min_cells=min_num_cells_per_gene)
