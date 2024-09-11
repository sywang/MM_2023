import warnings
from typing import Dict, Optional, List, Callable

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import ClusterMixin
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm


def clustering_stability_search(X: pd.DataFrame, clustering_models: Dict[str, ClusterMixin],
                                clustering_metrics: Optional[List[Callable]] = None,
                                n_iter=50, f=0.9) -> pd.DataFrame:
    if clustering_metrics is None:
        clustering_metrics = [metrics.adjusted_rand_score]

    all_results = {}
    for model_name, model in tqdm(clustering_models.items()):
        model_results = []
        for i in range(n_iter):
            sample_size = int(len(X) * f)
            sample_ids = list(X.index)
            sample_1 = np.random.choice(sample_ids, size=sample_size, replace=False)
            sample_2 = np.random.choice(sample_ids, size=sample_size, replace=False)

            common_sample = list(set(sample_1).intersection(set(sample_2)))

            X_sub_1 = X.loc[sample_1]
            X_sub_2 = X.loc[sample_2]
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    y_sub_1 = pd.Series(model.fit_predict(X_sub_1), index=sample_1)
                    y_sub_2 = pd.Series(model.fit_predict(X_sub_2), index=sample_2)
                    metrics_list = []
                    for clustering_metric in clustering_metrics:
                        metric = clustering_metric(y_sub_1[common_sample], y_sub_2[common_sample])
                        if isinstance(metric, tuple):  # v_measure returns a tuple
                            metric = metric[-1]

                        metrics_list.append(metric)
                    model_results.append(tuple(metrics_list))

                except ConvergenceWarning:
                    model_results.append(-1)

        all_results[model_name] = model_results
    return pd.DataFrame(all_results)
