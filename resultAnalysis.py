# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 02:01:59 2021

@author: hlinl
"""

import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io
import shutil
import multiprocessing
import sklearn.cluster as skc
import sklearn.metrics as skm

from sklearn import manifold
from mpl_toolkits import mplot3d
from matplotlib.ticker import NullFormatter

"""
Result post-analysis. 
"""


DEBUG = True


class DBSCAN():
    """
    """
    
    def __init__(self, data_matrix, file_paths_list):
        """
        Class initialization. 
        
        Parameters:
        ----------
        data_matrix : 2D Float array. 
            The matrix read from the image file.

        Returns:
        -------
        None.
        """
        
        self._epsilon = 0.02 # Default: 0.01. 0.02.
        self._minPts = 10 # Default: 3.
        
        self.data_matrix = data_matrix
        self.file_paths_list = file_paths_list
        self.fileName_index_dict = {}
        self.index_fileName_dict = {}
        self._cluster_dict = {}
        self._cluster_list_dict = {}
        self._dist_matrix = None
        
        self._clusterNum = None # Exclude noise. 
        self._colorList = ['r', 'g', 'b', 'c', 'm', 'y']
        self._markerList = ['.']

    
    def _clusterDictEstablishment(self):
        """
        Create the cluster dictionary. 
        
        Parameters:
        ----------
        None. 

        Returns:
        -------
        None.
        """
        
        for ind, file in enumerate(self.file_paths_list):
            file_name = file.split('.')[0]
            self.fileName_index_dict[file_name] = ind
            self.index_fileName_dict[ind] = file_name
            self._cluster_dict[ind] = None
        

    def _getNeighbors(self, ind):
        """
        Find neighbor pixels with the given label. 
        
        Parameters:
        ----------
            ind_tuple : Int tuple. 
                The coordinate of center point. Format: (row, col).     

        Returns:
        -------
            neighbor_list: List of tuples. 
                The list of found neighbor coordinates. 
        """
        
        search_array = copy.deepcopy(self._dist_matrix[ind,:])
        search_array[search_array > self._epsilon] = -1
        neighbor_list = [i for i, val in enumerate(search_array) if val >= 0]
        
        return neighbor_list
    
    
    def _compute_pairwiseDistance(self, matrix, axis):
        """
        Axis: 0 or 1. 
        """
        
        n = matrix.shape[axis]
        
        if axis == 0: G = np.dot(matrix, matrix.T)
        else: G = np.dot(matrix.T, matrix)
        
        H = np.tile(np.diag(G), (n,1))
        
        return np.sqrt(H + H.T - 2*G)
    
    
    def clustering(self):
        """
        """
        
        self._clusterDictEstablishment()
        self._dist_matrix = self._compute_pairwiseDistance(self.data_matrix, axis=1)
        
        cluster_num = 0
        for ki in self._cluster_dict.keys():
            if self._cluster_dict[ki] != None: continue
            neighbor_list_temp = self._getNeighbors(ki)
            
            if len(neighbor_list_temp) < self._minPts: self._cluster_dict[ki] = 0
            else:
                cluster_pts_list = []
                cluster_num += 1
                cluster_pts_list.append(ki)
                cluster_pts_list = self._clusterExpansion(cluster_pts_list, neighbor_list_temp)
        
                for pt in cluster_pts_list: self._cluster_dict[pt] = cluster_num
        
        for i in range(cluster_num+1):
            self._cluster_list_dict[i] = copy.deepcopy(list(set([key for (key, val) 
                                                                 in self._cluster_dict.items() 
                                                                 if val == i])))
        
        self._clusterNum = cluster_num
    
    
    def _clusterExpansion(self, cluster_pts_list, neighbor_list):
        """
        """
        
        cluster_size, new_cluster_pts_list = len(cluster_pts_list), []
        for item in neighbor_list:
            if self._cluster_dict[item] == None or self._cluster_dict[item] == 0:
                cluster_pts_list.append(item)
            else: continue
            
            new_neighbor_list = self._getNeighbors(item)
            
            if len(new_neighbor_list) < self._minPts: continue
            else: new_cluster_pts_list += list(set(new_neighbor_list).difference(cluster_pts_list))
        
        cluster_pts_list = list(set(cluster_pts_list))
        new_cluster_pts_list = list(set(new_cluster_pts_list))
        
        if len(cluster_pts_list) > cluster_size: 
            return self._clusterExpansion(cluster_pts_list, new_cluster_pts_list)
        else: return cluster_pts_list


def normalization(matrix, axis):
    """
    """
    
    min_vect = np.min(matrix, axis=axis).reshape(-1,1)
    max_vect = np.max(matrix, axis=axis).reshape(-1,1)
    
    matrix_normalized = (matrix - min_vect) / (max_vect - min_vect) * 2. - 1.
    
    return matrix_normalized


def standardization(matrix, axis):
    """
    """
    
    mean_vect = np.mean(matrix, axis=axis).reshape(-1,1)
    std_vect = np.std(matrix, axis=axis).reshape(-1,1)
    
    matrix_standardized = (matrix - mean_vect) / std_vect
    
    return matrix_standardized


def zeroMean(matrix):
    """
    """
    
    mean_vect = np.mean(matrix, axis=1)
    matrix_new = np.zeros(shape=matrix.shape)
    
    for i in range(matrix.shape[1]):
        matrix_new[:,i] = matrix[:,i] - mean_vect

    return matrix_new, mean_vect


def PCA(matrix, PC_num):
    """
    Implement PCA on the input matrix and return PC_num weights and eigenvectors. 
    
    Parameters:
    ----------
        matrix: 2D matrix. The target matrix. Size: 2 x N. 
        PC_num: Int. The number of principal components. 
        
    Returns: 
    ----------
        eigVect: 2D Float matrix. The matrix of eigenvectors (sorted).  
        eigVal: 1D Float array. The vector of eigenvalues (sorted). 
        weights: 2D Float diagonal matrix. The weights of principal components. 
    """
    
    matrix, mean_vect = zeroMean(matrix)
    cov_matrix = matrix @ np.transpose(matrix) # Size: 2 x 2. 
    eigVal_full, eigVect_full = np.linalg.eig(cov_matrix)
    
    # PCA
    eigVal = np.zeros(shape=(PC_num, 1), dtype=complex)
    eigVect = np.zeros(shape=(eigVect_full.shape[0], PC_num), dtype=complex)
    eigVal_sorted_indices = np.argsort(np.real(eigVal_full))
    eigVal_PC_indices = eigVal_sorted_indices[-1:-(PC_num+1):-1] # Pick PC_num indices of largest principal eigenvalues
    
    for i, index in enumerate(eigVal_PC_indices): # From biggest to smallest. 
        eigVal[i,0] = eigVal_full[index] # Pick PC_num principal eigenvalues. Sorted. 
        eigVect[:,i] = eigVect_full[:,index] # Pick PC_num principal eigenvectors. Sorted. 
    
    # Compute weights of each sample on the picked basis (encoding). 
    eigVal, eigVect = np.real(eigVal), np.real(eigVect)
    weights = np.transpose(eigVect) @ matrix # Size: PC_num * SampleNum. 
    
    return eigVect, eigVal, weights, mean_vect


def getObjectArray(PCA_invariants_array, Hu_invariants_array, 
                   features_PCA_ind_list, features_Hu_ind_list):
    """
    """
    
    dim_1 = PCA_invariants_array.shape[1]
    PCA_array_part = np.array([]).reshape(0,dim_1)
    Hu_array_part = np.array([]).reshape(0,dim_1)
    
    for ind_temp in features_PCA_ind_list:
        PCA_feature_array_temp = PCA_invariants_array[ind_temp,:]
        PCA_array_part = np.vstack((PCA_array_part, PCA_feature_array_temp))
        
    for ind_temp in features_Hu_ind_list:
        Hu_feature_array_temp = Hu_invariants_array[ind_temp,:]
        Hu_array_part = np.vstack((Hu_array_part, Hu_feature_array_temp))
    
    return np.vstack((PCA_array_part, Hu_array_part))


def clustering(data_matrix, method, file_paths_list=[], n_clusters=2):
    """
    method: "DBSCAN" or "KMEANS". 
    n_clusters: only for k-means. 
    file_paths_list: only for DBSCAN. 
    """
    
    if method == "DBSCAN":
        dbscan = DBSCAN(data_matrix, file_paths_list)
        dbscan.clustering()
        return dbscan
    
    if method == "KMEANS":
        kmeans = skc.KMeans(n_clusters, random_state=0).fit(data_matrix.T)
        return kmeans


def parameterization_KMeans(data_matrix, n_clusters_range, figure_name):
    """
    """
    
    n_clusters_list, mean_min_dist_list = [], []
    for n_clusters in range(n_clusters_range[0], n_clusters_range[1]+1):
        kmeans_temp = skc.KMeans(n_clusters, random_state=0).fit(data_matrix.T)
        dist_matrix_temp = kmeans_temp.transform(data_matrix.T)
        
        min_dist_array_temp = np.min(dist_matrix_temp, axis=1)
        mean_min_dist_temp = np.mean(min_dist_array_temp)
        
        mean_min_dist_list.append(mean_min_dist_temp)
        n_clusters_list.append(n_clusters)
        
    plt.figure(figsize=(20,20))
    plt.rcParams.update({"font.size": 35})
    plt.tick_params(labelsize=35)
    plt.plot(n_clusters_list, mean_min_dist_list, c='b', linewidth=3.0, 
             marker='v', markersize=15.0, label="Mean_min_dist")
    plt.xlabel("Number of clusters", fontsize=40)
    plt.ylabel("Average minimum distance", fontsize=40)
    plt.legend(loc="upper right", prop={"size": 40})
    plt.savefig(figure_name)


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def tSNE_plot(data, label_array, n_components, figure_name="tSNE_plot.png"):
    """
    """
    
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    Y = tsne.fit_transform(data)
    
    colors_map_list = ['red', 'green', 'blue', 'orange', 'purple', 
                       'pink', 'gray', 'cyan', 'brown', 'olive']
    color_plot_list = []
    for i in range(data.shape[0]):
        color_plot_list.append(colors_map_list[label_array[i] % len(colors_map_list)])
    
    plt.figure(figsize=(20,20))
    if n_components == 2:
        plt.scatter(Y[:,0], Y[:,1], c=color_plot_list, cmap=plt.cm.Spectral, linewidths=1.0)
        plt.gca().set_aspect('equal', adjustable='box')
    if n_components == 3:
        ax = plt.gca(projection='3d')
        ax.scatter(Y[:,0], Y[:,1], Y[:,2], c=color_plot_list, cmap=plt.cm.Spectral, linewidths=1.0)
        set_axes_equal(ax)
        ax.view_init(-140, 30)
        
    plt.savefig(figure_name)


def PCA_visualization_plot(data, label_array, n_components, figure_name="PCA_plot.png"):
    """
    """
    
    eigVect, eigVal, weights, mean_vect = PCA(data, PC_num=n_components)
    
    colors_map_list = ['red', 'green', 'blue', 'orange', 'purple', 
                       'pink', 'gray', 'cyan', 'brown', 'olive']
    color_plot_list = []
    for i in range(data.shape[1]):
        color_plot_list.append(colors_map_list[label_array[i] % len(colors_map_list)])
    
    plt.figure(figsize=(8,8))
    if n_components == 2:
        plt.scatter(weights[0,:], weights[1,:], c=color_plot_list, 
                    cmap=plt.cm.Spectral, linewidths=1.0)
        plt.gca().set_aspect('equal', adjustable='box')
    if n_components == 3:
        ax = plt.gca(projection='3d')
        ax.scatter(weights[0,:], weights[1,:], weights[2,:], c=color_plot_list, 
                   cmap=plt.cm.Spectral, linewidths=1.0)
        set_axes_equal(ax)
        ax.view_init(-140, 30)
        
    plt.savefig(figure_name)


def raw_visualization_plot(data, label_array, n_features, figure_name="raw_visualization_plot.png"):
    """
    Data: should be object feature array. 
    n_features: 2 or 3. 
    """
    
    colors_map_list = ['red', 'green', 'blue', 'orange', 'purple', 
                       'pink', 'gray', 'cyan', 'brown', 'olive']
    color_plot_list = []
    for i in range(data.shape[1]):
        color_plot_list.append(colors_map_list[label_array[i] % len(colors_map_list)])
    
    plt.figure(figsize=(8,8))
    if n_features == 1:
        plt.plot(data[0,:], data[0,:], c='b', linewidth=3.0, 
                 marker='v', markersize=15.0, label="Mean_min_dist")
    if n_features == 2:
        plt.scatter(data[0,:], data[1,:], c=color_plot_list, cmap=plt.cm.Spectral, linewidths=1.0)
        plt.gca().set_aspect('equal', adjustable='box')
    if n_features == 3:
        ax = plt.gca(projection='3d')
        ax.scatter(data[0,:], data[1,:], data[2,:], c=color_plot_list, 
                   cmap=plt.cm.Spectral, linewidths=1.0)
        set_axes_equal(ax)
        ax.view_init(-140, 30)
    
    plt.savefig(figure_name)


def evaluateClusterGoodness(data, label_array):
    """
    """
        
    silhouette_avg = skm.silhouette_score(data, label_array)
    sample_silhouette_values = skm.silhouette_samples(data, label_array)

    return silhouette_avg, sample_silhouette_values


def clusterGoodness_Visualization(data, label_array, center_array, n_clusters,
                                  figure_name="cluster_goodness_eval_visualization.png"):
    """
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(72, 28)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, data.shape[0] + (n_clusters + 1) * 10])

    silhouette_avg, sample_silhouette_values = evaluateClusterGoodness(data, label_array)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[label_array == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(label_array.astype(float) / n_clusters)
    ax2.scatter(data[:, 0], data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Draw white circles at cluster centers
    ax2.scatter(center_array[:, 0], center_array[:, 1], marker='o',
                c="white", alpha=1, s=500, edgecolor='k')

    for i, c in enumerate(center_array):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                  fontsize=40, fontweight='bold')
    plt.savefig(figure_name)
    
    return silhouette_avg


def kmeans_ClusterNum_parametric_study(data, kmeans_n_clusters_list):
    """
    """
    
    silhouette_avg_list = []
    for kmeans_n_clusters_temp in kmeans_n_clusters_list:
        kmeans = clustering(data, method="KMEANS", n_clusters=kmeans_n_clusters_temp)
        label_array = kmeans.labels_
        center_array = kmeans.cluster_centers_
        
        silhouette_avg_temp = clusterGoodness_Visualization(data.T, label_array, center_array, kmeans_n_clusters_temp, 
                                                            figure_name="cluster_goodness_eval_visualization_{}.png".format(kmeans_n_clusters_temp))
        silhouette_avg_list.append(silhouette_avg_temp)
        
        print("Cluster Number: {} | Evaluation Score: {}".format(kmeans_n_clusters_temp, silhouette_avg_temp))
    
    return silhouette_avg_list


if __name__ == "__main__":
    # ===================== Initial parameters ===================== # 
    file_directory = "result_invariants_archive"
    image_directory = "images"
    file_paths_list = os.listdir(file_directory)
    result_directory = "result_analysis"
    clustering_method = "KMEANS" # "DBSCAN" or "KMEANS". 
    
    kmeans_n_clusters = 10 # 7, 8, 10, 15, 20
    
    isNorm_PCA, isStandard_PCA = False, True
    isNorm_Hu, isStandard_Hu = False, True
    isScaleInvariant = False
    
    isVisualize = True
    visualization_space_dimensions = 2 # 2 or 3.
    
    features_PCA_ind_list = [2,3] # [2].
    features_Hu_ind_list = [] # [], [0,1], [0,1,2,3], [0,1,2,3,4,5,6].
    
    
    # ===================== Data modification, normalization & standardization ===================== # 
    for ind, file in enumerate(file_paths_list):
        mdict = scipy.io.loadmat(os.path.join(file_directory, file))
        image_PCA_invariants_array_temp = mdict["image_PCA_invariants_array"].reshape(-1,1)
        image_Hu_invariants_array_temp = mdict["image_Hu_invariants_array"].reshape(-1,1)
        
        # ======================== Remove Hu scale independency ====================== #
        
        if not isScaleInvariant:
            image_matrix_temp, indices_array_temp = mdict["image_matrix"], mdict["indices_array"]
            
            Mu_00_temp = 0
            for k in range(indices_array_temp.shape[1]):
                i_temp, j_temp = indices_array_temp[0,k], indices_array_temp[1,k]
                Mu_00_temp += image_matrix_temp[i_temp,j_temp]
            
            scaling_array = np.array([Mu_00_temp**2, Mu_00_temp**4, Mu_00_temp**5, 
                                      Mu_00_temp**5, Mu_00_temp**10, Mu_00_temp**7, 
                                      Mu_00_temp**10]).reshape(-1,1) # Make it scale-dependent. 
            
            image_Hu_invariants_array_temp = image_Hu_invariants_array_temp * scaling_array
            
        # ============================================================================ #
        
        if ind == 0:
            image_PCA_invariants_array_stack = image_PCA_invariants_array_temp
            image_Hu_invariants_array_stack = image_Hu_invariants_array_temp
        else:
            image_PCA_invariants_array_stack = np.hstack((image_PCA_invariants_array_stack,
                                                          image_PCA_invariants_array_temp))
            image_Hu_invariants_array_stack = np.hstack((image_Hu_invariants_array_stack,
                                                          image_Hu_invariants_array_temp))
    
    if isNorm_Hu:
        image_Hu_invariants_array_stack = normalization(image_Hu_invariants_array_stack, axis=1)
    if isStandard_Hu:
        image_Hu_invariants_array_stack = standardization(image_Hu_invariants_array_stack, axis=1)
    if isNorm_PCA:
        image_PCA_invariants_array_stack = normalization(image_PCA_invariants_array_stack, axis=1)
    if isStandard_PCA:
        image_PCA_invariants_array_stack = standardization(image_PCA_invariants_array_stack, axis=1)
    
    
    # ===================== Feature array construction ===================== #
    object_array_stack = getObjectArray(image_PCA_invariants_array_stack, 
                                        image_Hu_invariants_array_stack, 
                                        features_PCA_ind_list, features_Hu_ind_list)
    
    
    # ===================== K-Means parametric study ===================== #
    if clustering_method == "KMEANS":
        kmeans_n_clusters_range = (3, 20) # Inclusive at both ends. 
        kmeans_n_clusters_list = list(np.arange(kmeans_n_clusters_range[0],
                                                kmeans_n_clusters_range[1]+1))
        silhouette_avg_list = kmeans_ClusterNum_parametric_study(object_array_stack, 
                                                                 kmeans_n_clusters_list)
        
        plt.figure(figsize=(20,20))
        plt.rcParams.update({"font.size": 35})
        plt.tick_params(labelsize=35)
        plt.plot(kmeans_n_clusters_list, silhouette_avg_list, c='b', linewidth=3.0, 
                 marker='v', markersize=15.0, label="Silhouette_score")
        plt.xlim(kmeans_n_clusters_range)
        plt.xlabel("Number of clusters", fontsize=40)
        plt.ylabel("Silhouette score", fontsize=40)
        plt.legend(loc="upper right", prop={"size": 40})
        plt.savefig("nClusters_vs_Silhouette_plot.png")
    
    
    # ===================== Clustering ===================== #
    kmeans_n_clusters = kmeans_n_clusters_list[silhouette_avg_list.index(max(silhouette_avg_list))]
    
    if clustering_method == "DBSCAN": 
        dbscan = clustering(object_array_stack, clustering_method, 
                            file_paths_list=file_paths_list)
        
        label_list = []
        for ind, file_name in enumerate(file_paths_list):
            label_list.append(dbscan._cluster_dict[ind])
        
        label_array = np.array(label_list).astype(int)
        dist_matrix = dbscan._dist_matrix
        
    if clustering_method == "KMEANS":
        kmeans = clustering(object_array_stack, clustering_method, 
                            n_clusters=kmeans_n_clusters)
        label_array = kmeans.labels_
        center_array = kmeans.cluster_centers_
    
    
    # ===================== Data visualization ===================== #
    n_features_PCA, n_features_Hu = len(features_PCA_ind_list), len(features_Hu_ind_list)
    if visualization_space_dimensions > n_features_PCA + n_features_Hu: 
        isVisualize = False
    
    if isVisualize:
        tSNE_plot(object_array_stack.T, label_array, n_components=visualization_space_dimensions) # t-SNE visualization of high-dimensional data. 
        PCA_visualization_plot(object_array_stack, label_array, n_components=visualization_space_dimensions) # PCA visualization of high-dimensional data.
        raw_visualization_plot(object_array_stack, label_array, n_features=visualization_space_dimensions) # Raw visualization of object feature array. 
    
        
    # ===================== Evaluate clustering goodness ===================== #
    silhouette_avg = clusterGoodness_Visualization(object_array_stack.T, label_array, 
                                                   center_array, kmeans_n_clusters)
        
    
    # ===================== Remove previous files =====================#
    for file in os.listdir(result_directory):
        shutil.rmtree(os.path.join(result_directory, file))
    
    
    # ===================== Collect image files into separate cluster folders ===================== #
    for i, file in enumerate(file_paths_list):
        file_name = file.split('.')[0]
        
        if clustering_method == "DBSCAN":
            ind_temp = dbscan.fileName_index_dict[file_name]
            label_temp = dbscan._cluster_dict[ind_temp]
        if clustering_method == "KMEANS":
            label_temp = label_array[i]
        
        image_path = os.path.join(image_directory, "{}.png".format(file_name))
        cluster_path = os.path.join(result_directory, str(label_temp))
        
        if not os.path.isdir(cluster_path): os.mkdir(cluster_path)
        
        shutil.copy(image_path, os.path.join(cluster_path, "{}.png".format(file_name)))
        
        # print("File: {}.png categorized.".format(file_name))
    
    
    # ===================== K-Means parametric study ===================== #
    if clustering_method == "KMEANS": 
        parameterization_KMeans(object_array_stack, n_clusters_range=(1,100), 
                                figure_name="kmeans_parameterization.png")

    
        
