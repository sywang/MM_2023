import re
from typing import List

import anndata as ad


def drop_blacklist_genes_by_prefix(adata: ad.AnnData, blacklist_genes_prefixes: List[str]) -> ad.AnnData:
    blacklist_genes_regexs = [f"^{prefix}*" for prefix in blacklist_genes_prefixes]
    adata = adata[:, [gn for gn in adata.var_names if re.match("|".join(blacklist_genes_regexs), gn)]]
    return adata
