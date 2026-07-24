import math
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from typing import Any, Callable, List
from utilities.covariance_utilities import covariance_to_correlation
from utilities.hierarchical_clustering import hierarchical_clustering
from utilities.principal_component_analysis import detone


def correlation_to_hrp_distance(correlation: np.ndarray) -> np.ndarray:
    """
    Convert a correlation matrix to the Lopez de Prado HRP distance matrix
    ``d_{i,j} = sqrt(0.5 * (1 - rho_{i,j}))``.

    The result is a proper metric (see lab notes Question 3) and is the input
    expected by the HRP clustering step.

    Parameters:
        correlation (np.ndarray): Correlation matrix.

    Returns:
        np.ndarray: Distance matrix with the same shape as ``correlation``.
    """

    correlation = np.clip(correlation, -1.0, 1.0)
    return np.sqrt(0.5 * (1.0 - correlation)) # ERROR: IT IS NOT 2, BUT 0.5, and it was + and not -


def flatten_list(lst: List[Any]) -> List[Any]:
    """
    Recursively flatten a nested list into a single list of elements.

    Parameters:
        lst (List[Any]): The nested list.

    Returns:
        List[Any]: A flattened list of elements.
    """

    if isinstance(lst, list):
        return [item for sub_lst in lst for item in flatten_list(sub_lst)]
    else:
        return [lst]


def is_nested(lst: List[Any]) -> bool:
    """
    Check if a list is nested (contains other lists as elements).

    Parameters:
        lst (List[Any]): The list to check.

    Returns:
        bool: True if the list is nested, False otherwise.
    """

    return any(isinstance(element, list) for element in lst)


def list_recursive_bisection(
    lst: List[Any],
    labels: List[Any] | None = None,
    cur_iter: int | None = None,
    max_iter: int | None = None,
) -> List[Any]:
    """
    Perform a recursive bisection of a list.

    Parameters:
        lst (List[Any]): The original list to be bisected.
        labels (List[Any] | None): Labels, default to None.
        cur_iter (int | None): Current iteration, default to None.
        max_iter (int | None): Maximum number of iterations, default to None.

    Returns:
        list: A nested list representing the recursive bisection.
    """
    # Initialization
    if cur_iter is None:
        cur_iter = 0
        
    # Cases in which we return the result and we do not do the recursion 
    if len(lst) <= 1 or (max_iter is not None and cur_iter >= max_iter):
        if labels is not None:
            return [labels[i] for i in lst]
        return lst

    # Bisection
    mid = len(lst) // 2
    
    # Recursion on both the two arrays created
    left_cluster = list_recursive_bisection(
        lst[:mid], labels=labels, cur_iter=cur_iter + 1, max_iter=max_iter
    )
    
    right_cluster = list_recursive_bisection(
        lst[mid:], labels=labels, cur_iter=cur_iter + 1, max_iter=max_iter
    )
    
    # Return the list
    return [left_cluster, right_cluster]


def recursive_bisection(
    linkage_matrix: pd.DataFrame,
    labels: List[Any] | None = None,
    clusters_num: int | None = None,
) -> List[Any]:
    """
    Return the nested cluster structure from a linkage matrix performing a recursive bisection.

    Parameters:
        linkage_matrix (pd.DataFrame): Linkage matrix.
        labels (List[Any] | None): Labels, default to None.
        clusters_num (int | None): Number of clusters to be formed, default to None, i.e. use the
            whole linkage matrix.

    Returns:
        List[Any]: Nested clusters.
    """

    iter_num = None if clusters_num is None else math.log(clusters_num, 2)
    if iter_num is not None:
        if not iter_num.is_integer():
            raise ValueError(
                "The number of clusters must be a power of 2 for recursive bisection."
            )
        else:
            iter_num = int(iter_num)

    leaves_lst = sch.leaves_list(linkage_matrix.values)
    leaves_labels = (
        list(leaves_lst) if labels is None else [labels[leaf] for leaf in leaves_lst]
    )

    nested_clusters = list_recursive_bisection(leaves_labels, max_iter=iter_num)
    return  nested_clusters


def dendrogram_iteration(
    linkage_matrix: pd.DataFrame,
    labels: List[Any] | None = None,
    clusters_num: int | None = None,
) -> List[Any]:
    """
    Convert a linkage matrix to nested clusters according to the dendrogram structure.

    Parameters:
        linkage_matrix (pd.DataFrame): Linkage matrix.
        labels (List[Any] | None): Labels, default to None.
        clusters_num (int | None): Number of clusters to be formed, default to None, i.e. use the
            whole linkage matrix.

    Returns:
        List[Any]: Nested clusters.
    """
    # Initialize linkage matrix
    Z = linkage_matrix.values if isinstance(linkage_matrix, pd.DataFrame) else linkage_matrix
    N = Z.shape[0] + 1  # number of assets
    
    if labels is None:
        labels = list(range(N))
        
    # Nodes in which we stop fragmentation
    if clusters_num is None or clusters_num >= N:
        # primo caso: We don't stop
        target_nodes = set(range(N))
    else:
        # altro caso: we start from the last cluster
        active_nodes = {2 * N - 2}
        
        # We open the aggregate as last since it is the one at more distance
        while len(active_nodes) < clusters_num:
            split_node = max(active_nodes)
            active_nodes.remove(split_node)
            
            left_child = int(Z[split_node - N, 0])
            right_child = int(Z[split_node - N, 1])
            
            active_nodes.add(left_child)
            active_nodes.add(right_child)
            
        target_nodes = active_nodes

    # bottom-up
    # Initialize for bottom leaves
    leaves_dict = {i: [labels[i]] for i in range(N)}
    structure_dict = {i: [labels[i]] for i in range(N)}
    
    # linkage matrix from the bottom to the top
    for i in range(N - 1):
        current_node = N + i
        left_child = int(Z[i, 0])
        right_child = int(Z[i, 1])
        
        # Update flat leaves aggregating the sons leaves
        leaves_dict[current_node] = leaves_dict[left_child] + leaves_dict[right_child]
        
        # Compute the annidated structure
        if current_node in target_nodes:
            # Caso 1: cluster target
            structure_dict[current_node] = leaves_dict[current_node]
        else:
            structure_dict[current_node] = [
                structure_dict[left_child], 
                structure_dict[right_child]
            ]
            
    # root node
    root_node = 2 * N - 2
    return structure_dict[root_node]


def top_down_allocation(
    nested_clusters: List[Any], covariance: pd.DataFrame, get_cluster_var: Callable
) -> pd.Series:
    """
    Top-down allocation of weights to the clusters following Lopez de Prado's
    recursive split: at each bisection ``alpha = sigma2_R / (sigma2_L + sigma2_R)``
    is allocated to the left cluster, ``1 - alpha`` to the right one (more capital
    to the lower-variance branch). The variance of each cluster is computed via
    ``get_cluster_var``.

    Parameters:
        nested_clusters (List[Any]): The nested cluster structure produced by
            ``recursive_bisection`` or ``dendrogram_iteration``.
        covariance (pd.DataFrame): Covariance matrix.
        get_cluster_var (Callable): Function returning the variance of a cluster
            given the covariance matrix and the list of asset labels in the
            cluster.

    Returns:
        pd.Series: Weights summing to 1, indexed on the asset labels.
    """

    weights = pd.Series(1.0, index=flatten_list(nested_clusters))
    if not is_nested(nested_clusters):
        return weights
    else:
        cluster1 = flatten_list(nested_clusters[0])
        cluster2 = flatten_list(nested_clusters[1])

        cluster1_var = get_cluster_var(covariance=covariance, cluster=cluster1)
        cluster2_var = get_cluster_var(covariance=covariance, cluster=cluster2)

        # Inverse volatility between clusters
        alpha1 = cluster2_var / (cluster1_var + cluster2_var)   # Error: the initial formula was totally wrong
        alpha2 = 1 - alpha1                                     # Error: the initial relation with alpha 1 was wrong

        weights[cluster1] *= alpha1 * top_down_allocation(
            nested_clusters[0], covariance.loc[cluster1, cluster1], get_cluster_var
        )
        weights[cluster2] *= alpha2 * top_down_allocation(
            nested_clusters[1], covariance.loc[cluster2, cluster2], get_cluster_var
        )

        return weights


def get_cluster_var_via_iv(covariance: pd.DataFrame, cluster: List[Any]):

    covariance = covariance.loc[cluster, cluster]
    iv_weights = 1.0 / np.diag(covariance)
    iv_weights /= iv_weights.sum()
    iv_weights = iv_weights.reshape(-1, 1)

    return (iv_weights.T @ covariance.values @ iv_weights)[0, 0]


def hierarchical_risk_parity(
    covariance: pd.DataFrame,
    linkage_method: str = "single",
    distance_metric: str = "euclidean",
    cluster_traverser: Callable = dendrogram_iteration,
    get_cluster_var: Callable = get_cluster_var_via_iv,
    perform_detoning: bool = False,
    plot_dendrogram: bool = False,
) -> pd.Series:
    """
    Compute Hierarchical Risk Parity weights for a single rebalance.

    Pipeline:
        1. Convert ``covariance`` to a correlation matrix.
        2. Optionally detone the correlation matrix by removing its first
           principal component.
        3. Build a linkage matrix on the HRP distance ``sqrt(0.5(1 - rho))``
           using the chosen linkage method and (second-stage) distance metric.
        4. Convert the linkage to a nested cluster structure via
           ``cluster_traverser`` (e.g. ``recursive_bisection`` or
           ``dendrogram_iteration``).
        5. Allocate weights top-down with inverse-variance splits.

    Parameters:
        covariance (pd.DataFrame): Covariance matrix indexed by asset labels.
        linkage_method (str): Linkage method passed to ``scipy.linkage``
            (``single``, ``ward``, ``complete``, ``average``, ...).
        distance_metric (str): Metric used by ``scipy.linkage`` to compute
            distances between rows of the input matrix (defaults to
            ``euclidean``, following Lopez de Prado).
        cluster_traverser (Callable): Function mapping a linkage matrix to a
            nested cluster structure. Defaults to ``dendrogram_iteration``.
        get_cluster_var (Callable): Function returning the variance of a
            cluster. Defaults to inverse-variance parity on the diagonal.
        perform_detoning (bool): Remove the first principal component from the
            correlation matrix before clustering. Defaults to False.
        plot_dendrogram (bool): Whether to plot the dendrogram while building it.

    Returns:
        pd.Series: HRP weights summing to 1, indexed on the asset labels.
    """
    correlation = covariance_to_correlation(covariance=covariance.values)
    if perform_detoning:
        correlation = detone(corr_matrix=correlation, components_num=1)
    distance = correlation_to_hrp_distance(correlation)
    linkage_matrix = hierarchical_clustering(
        matrix=pd.DataFrame(
            distance, index=covariance.index, columns=covariance.columns
        ),
        linkage_method=linkage_method,
        distance_metric=distance_metric,
        plot_dendrogram=plot_dendrogram,
    )

    nested_clusters = cluster_traverser(
        linkage_matrix, labels=covariance.index.tolist()
    )

    return top_down_allocation(
        nested_clusters=nested_clusters,
        covariance=covariance,
        get_cluster_var=get_cluster_var,
    )
