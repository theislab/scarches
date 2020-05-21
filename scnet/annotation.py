from collections import Counter

import numpy as np
from sklearn.neighbors import KNeighborsTransformer

from scnet.utils import remove_sparsity


def weighted_knn(train_adata, valid_adata, label_key, n_neighbors=50, threshold=0.5,
                 pred_unknown=True, return_uncertainty=True):
    print(f'Weighted KNN with n_neighbors = {n_neighbors} and threshold = {threshold} ... ', end='')
    k_neighbors_transformer = KNeighborsTransformer(n_neighbors=n_neighbors, mode='distance',
                                                    algorithm='brute', metric='euclidean',
                                                    n_jobs=-1)
    train_adata = remove_sparsity(train_adata)
    valid_adata = remove_sparsity(valid_adata)

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
        labels = y_train_labels[top_k_indices[i]]
        most_common_label, _ = Counter(y_train_labels[top_k_indices[i]]).most_common(n=1)[0]
        most_prob = weights[i, y_train_labels[top_k_indices[i]] == most_common_label].sum()
        if pred_unknown:
            if most_prob >= threshold:
                pred_label = most_common_label
            else:
                pred_label = 'Unknown'
        else:
            pred_label = most_common_label

        if pred_label == y_valid_labels[i]:
            uncertainties.append(1 - most_prob)
        else:
            true_prob = weights[i, y_train_labels[top_k_indices[i]] == y_valid_labels[i]].sum()
            uncertainties.append(1 - true_prob)

        pred_labels.append(pred_label)

    pred_labels = np.array(pred_labels).reshape(-1, 1)
    uncertainties = np.array(uncertainties).reshape(-1, 1)

    print('finished!')
    if return_uncertainty:
        return pred_labels, uncertainties
    else:
        return pred_labels
