from collections import Counter

import numpy as np
from sklearn.neighbors import KNeighborsTransformer

from scarches.utils import remove_sparsity


def weighted_knn(train_adata, valid_adata, label_key, n_neighbors=50, threshold=0.5,
                 pred_unknown=True, mode='package'):
    """Annotates ``valid_adata`` cells with a trained weighted KNN classifier on ``train_adata``.

        Parameters
        ----------
        train_adata: :class:`~anndata.AnnData`
            Annotated dataset to be used to train KNN classifier with ``label_key`` as the target variable.
        valid_adata: :class:`~anndata.AnnData`
            Annotated dataset to be used to validate KNN classifier.
        label_key: str
            Name of the column to be used as target variable (e.g. cell_type) in ``train_adata`` and ``valid_adata``.
        n_neighbors: int
            Number of nearest neighbors in KNN classifier.
        threshold: float
            Threshold of uncertainty used to annotating cells as "Unknown". cells with uncertainties upper than this
             value will be annotated as "Unknown".
        pred_unknown: bool
            ``True`` by default. Whether to annotate any cell as "unknown" or not. If `False`, will not use
            ``threshold`` and annotate each cell with the label which is the most common in its
            ``n_neighbors`` nearest cells.
        mode: str
            Has to be one of "paper" or "package". If mode is set to "package", uncertainties will be 1 - P(pred_label),
            otherwise it will be 1 - P(true_label).
    """
    print(f'Weighted KNN with n_neighbors = {n_neighbors} and threshold = {threshold} ... ', end='')
    k_neighbors_transformer = KNeighborsTransformer(n_neighbors=n_neighbors, mode='distance',
                                                    algorithm='brute', metric='euclidean',
                                                    n_jobs=-1)
    k_neighbors_transformer.fit(train_adata.X)

    y_train_labels = train_adata.obs[label_key].values
    y_valid_labels = valid_adata.obs[label_key].values

    top_k_distances, top_k_indices = k_neighbors_transformer.kneighbors(X=valid_adata.X)

    stds = np.std(top_k_distances, axis=1)
    stds = (2. / stds) ** 2
    stds = stds.reshape(-1, 1)

    top_k_distances_tilda = np.exp(-np.true_divide(top_k_distances, stds))

    weights = top_k_distances_tilda / np.sum(top_k_distances_tilda, axis=1, keepdims=True)

    uncertainties = []
    pred_labels = []
    for i in range(len(weights)):
        unique_labels = np.unique(y_train_labels[top_k_indices[i]])
        best_label, best_prob = None, 0.0
        for candidate_label in unique_labels:
            candidate_prob = weights[i, y_train_labels[top_k_indices[i]] == candidate_label].sum()
            if best_prob < candidate_prob:
                best_prob = candidate_prob
                best_label = candidate_label
        
        if pred_unknown:
            if best_prob >= threshold:
                pred_label = best_label
            else:
                pred_label = 'Unknown'
        else:
            pred_label = best_label

        if mode == 'package':
            uncertainties.append(max(1 - best_prob, 0))

        elif mode == 'paper':
            if pred_label == y_valid_labels[i]:
                uncertainties.append(max(1 - best_prob, 0))
            else:
                true_prob = weights[i, y_train_labels[top_k_indices[i]] == y_valid_labels[i]].sum()
                if true_prob > 0.5:
                    pass
                uncertainties.append(max(1 - true_prob, 0))
        else:
            raise Exception("Invalid Mode!")

        pred_labels.append(pred_label)

    pred_labels = np.array(pred_labels).reshape(-1,)
    uncertainties = np.array(uncertainties).reshape(-1,)
    
    labels_eval = pred_labels == y_valid_labels
    labels_eval = labels_eval.astype(object)
    
    n_correct = len(labels_eval[labels_eval == True])
    n_incorrect = len(labels_eval[labels_eval == False]) - len(labels_eval[pred_labels == 'Unknown'])
    n_unknown = len(labels_eval[pred_labels == 'Unknown'])
    
    labels_eval[labels_eval == True] = f'Correct'
    labels_eval[labels_eval == False] = f'InCorrect'
    labels_eval[pred_labels == 'Unknown'] = f'Unknown'
    
    valid_adata.obs['uncertainty'] = uncertainties
    valid_adata.obs[f'pred_{label_key}'] = pred_labels
    valid_adata.obs['evaluation'] = labels_eval
    
    
    print('finished!')
    print(f"Number of correctly classified samples: {n_correct}")
    print(f"Number of misclassified samples: {n_incorrect}")
    print(f"Number of samples classified as unknown: {n_unknown}")