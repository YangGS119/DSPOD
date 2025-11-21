import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def HDM(datas, types):
    """
    Compute Hybrid Distance Metric.

    Parameters:
    -----------
    datas : np.ndarray
        Data matrix of shape (n_samples, m_features)
    types : np.ndarray
        Feature types array of shape (1, m_features) where 0 represents numerical and 1 represents nominal

    Returns:
    --------
    np.ndarray : Hybrid distance matrix of shape (n_samples, n_samples)
    """
    n, m = datas.shape

    num_fea = types[0] == 0
    nom_fea = types[0] == 1

    num_dis = squareform(pdist(datas[:, num_fea], metric="cityblock")) if num_fea.any() else np.zeros((n, n))
    nom_dis = squareform(pdist(datas[:, nom_fea], metric='hamming')) if nom_fea.any() else np.zeros((n, n))

    dis = num_dis + nom_dis * np.sum(nom_fea)

    return dis


def DSPOD(datas, types, K):
    """
    DSPOD (A Proximity Outlier Detection method fusing Density peaks and Structural information) algorithm.

    Parameters:
    -----------
    datas : np.ndarray
        Data matrix of shape (n_samples, m_features)
    types : np.ndarray
        Feature types array of shape (1, m_features) where 0 represents numerical and 1 represents nominal
    K : int
        Number of neighbors to consider

    Returns:
    --------
    np.ndarray : Outlier scores of shape (n_samples, 1)
    """
    n, m = datas.shape

    dist = HDM(datas, types)

    nbrs = NearestNeighbors(n_neighbors=K, leaf_size=10000, metric="precomputed").fit(dist)
    dis, index = nbrs.kneighbors(n_neighbors=K)

    dis = np.concatenate((np.zeros((n, 1)), dis), axis=1)
    index = np.concatenate((np.array(list(range(n)))[:, None], index), axis=1)

    lrd = np.zeros(n)
    for i in range(n):
        reach_dists = np.maximum(dis[i, 1:K + 1], dis[index[i, 1:K + 1], K])
        lrd[i] = 1 / (np.mean(reach_dists) + 1e-10)

    psdr = np.zeros(n)
    for i in range(n):
        neighbor_lrds = lrd[index[i, 1:K + 1]]
        npdd_lrd_idx = np.argmax(neighbor_lrds)
        npdd_lrd = neighbor_lrds[npdd_lrd_idx]
        psdr[i] = dis[i, npdd_lrd_idx + 1] * ((npdd_lrd / lrd[i]) - 1)

    e = 1 / (np.mean(1 - dis[:, 1:K + 1] / (dis[:, K].reshape(-1, 1) + 1e-10), axis=1) + 1e-10)

    s = np.zeros(n)
    for i in range(n):
        vectors = datas[index[i, 1:K + 1]] - datas[i]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        non_zero_mask = norms.flatten() > 0
        if np.any(non_zero_mask):
            vectors[non_zero_mask] /= norms[non_zero_mask]
        s[i] = np.linalg.norm(np.mean(vectors, axis=0))

    sdd = e * s
    PSOS = np.mean((psdr * sdd)[index[:, :K + 1]], axis=1)[:, None]

    return PSOS
