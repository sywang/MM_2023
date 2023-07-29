import logging

import anndata as ad
import scanpy as sc


def pp_drop_cells(adata, min_num_genes_per_cell: int, min_num_counts_per_cell: int,
                  mitochondrial_prefix: str, max_pct_mt_genes_pre_cell: float):
    logging.info("dropping bad cells")
    n_cells_before = len(adata.obs)

    sc.pp.filter_cells(adata, min_genes=min_num_genes_per_cell)
    sc.pp.filter_cells(adata, min_counts=min_num_counts_per_cell)

    n_cells_after_filter = len(adata.obs)
    logging.info(f"dropped by filter, dropped: {n_cells_before - n_cells_after_filter}")

    adata.var['mt'] = adata.var_names.str.startswith(mitochondrial_prefix)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < max_pct_mt_genes_pre_cell, :]
    n_cells_after_mt = len(adata.obs)
    logging.info(f"dropped by mitochondrial, dropped: {n_cells_after_filter - n_cells_after_mt}")

    n_cells_dropped = n_cells_before - n_cells_after_mt
    logging.info(f"dropped over all, before: {n_cells_before}, after: {n_cells_after_mt}, dropped: {n_cells_dropped}")


def pp_drop_genes(adata: ad.AnnData, min_num_cells_per_gene):
    logging.info("dropping bad genes")
    n_genes_before = len(adata.var)

    sc.pp.filter_genes(adata, min_cells=min_num_cells_per_gene)

    n_genes_after = len(adata.var)
    n_genes_dropped = n_genes_before - n_genes_after
    logging.info(f"dropped genes, before: {n_genes_before}, after: {n_genes_after}, dropped: {n_genes_dropped}")
