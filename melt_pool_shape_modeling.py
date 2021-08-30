import os
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mig


sys.setrecursionlimit(3000)
DEBUG = True


def indicesExtraction(data_matrix, pixel_value_thrsld):
    """
    Extract indices (row & col) of all points satisfying the pixel value threshold. 
    
    Parameters:
    ----------
        data_matrix: 2D Float Array, in [0.0, 1.0]. The matrix to be filetered. 
        pixel_value_threshold: Float tuple. The filter range for pixel values. 
    
    Returns:
    ----------
        indices_array: 2D Int array. The indices matrix of target pixels. Size: 2 x N. 
        mask_matrix: 2D Int array. Binary indicator of target pixels. Size: same as data_matrix. 
    """
    
    row_num, col_num = data_matrix.shape[0], data_matrix.shape[1]
    mask_matrix, indices_array = np.zeros(shape=data_matrix.shape), np.array([0,0]).reshape(-1,1)
    
    for i in range(row_num):
        for j in range(col_num):
            if (data_matrix[i,j] >= pixel_value_thrsld[0] and 
                data_matrix[i,j] < pixel_value_thrsld[1]):
                   indices_array_temp = np.array([j,i]).reshape(-1,1) # row and col swapped! 
                   mask_matrix[i,j] = 1 # Creating a binary matrix indicator. 
                   
                   if not indices_array.any(): indices_array = indices_array_temp
                   else: indices_array = np.hstack((indices_array, indices_array_temp))
    
    return mask_matrix, indices_array


class DBSCAN(object):
    """
    DBSCAN clustering implementation. 
    
    """
    
    def __init__(self, data_matrix):
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
        self.data_matrix = data_matrix
        self._intensity_range = (0.0, 1.0) # Default: (0.0, 1.0). 
        self._pixel_percentage_thrsld = [0.99, 1.0] # Percentage, sorted. Default: [0.99, 1.0]. Called in function "_clusterDictEstablishment". 
        self._pixel_intensity_thrsld = [0.9, 1.0] # Intensity, sorted. Default: [0.9, 1.0]. Called in function "_clusterDictEstablishment". 
        self._mask_matrix = np.zeros(shape=data_matrix.shape)
        self.indices_array = np.array([0,0]).reshape(-1,1) # Layout: (i,j). 
        self._cluster_dict = {}
        self._cluster_list_dict = {}
        self._epsilon = 1 # Default: 2. (2n+1)*(2n+1) grid search. 
        self._minPts = 8 # Default: 5. 
        self._clusterNum = None # Exclude noise. 
        self._colorList = ['r', 'g', 'b', 'c', 'm', 'y']
        self._markerList = ['.']
        
    
    def _indicesExtraction(self, pixel_value_thrsld):
        """
        Extract indices (row & col) of all points satisfying the pixel value threshold. 
        
        Parameters:
        ----------
            pixel_value_threshold: Float tuple. 
                The filter range for pixel values. 
        
        Returns:
        ----------
        None. 
        """
        
        row_num, col_num = self.data_matrix.shape[0], self.data_matrix.shape[1]
        
        for i in range(row_num):
            for j in range(col_num):
                for k in range(len(pixel_value_thrsld)-1):
                    if (self.data_matrix[i,j] >= pixel_value_thrsld[k] and 
                        self.data_matrix[i,j] <= pixel_value_thrsld[k+1]):
                           indices_array_temp = np.array([i,j]).reshape(-1,1)
                           self._mask_matrix[i,j] = k+1 # Creating a binary matrix indicator. 
                           
                           if not self.indices_array.any(): 
                               self.indices_array = indices_array_temp
                           else: 
                               self.indices_array = np.hstack((self.indices_array, 
                                                               indices_array_temp))
        
    
    def _indicesExtraction_IO(self, data_matrix, pixel_value_thrsld):
        """
        Extract indices (row & col) of all points satisfying the pixel value threshold. 
        
        Parameters:
        ----------
            data_matrix: 2D Float Array. 
                The matrix to be filtered. Range: [0.0, 1.0]. 
            pixel_value_threshold: Float tuple. 
                The filter range for pixel values. 
        
        Returns:
        ----------
            mask_matrix: 2D Int array. 
                Binary indicator of target pixels. Size: Same as data_matrix. 
            indices_array: 2D Int array. 
                The indices matrix of target pixels. Size: 2 x N.
            intensity_val_array: 1D Float array. 
                The array of intensity values of filtered pixels. 
            
        """
        
        row_num, col_num = data_matrix.shape[0], data_matrix.shape[1]
        mask_matrix, indices_array = np.zeros(shape=data_matrix.shape), np.array([0,0]).reshape(-1,1)
        intensity_val_array = []
        
        for i in range(row_num):
            for j in range(col_num):
                if (data_matrix[i,j] >= pixel_value_thrsld[0] and 
                    data_matrix[i,j] <= pixel_value_thrsld[1]):
                       indices_array_temp = np.array([i,j]).reshape(-1,1)
                       mask_matrix[i,j] = 1 # Creating a binary matrix indicator. 
                       
                       intensity_val_array.append(data_matrix[i,j])
                       if not indices_array.any(): indices_array = indices_array_temp
                       else: indices_array = np.hstack((indices_array, indices_array_temp))
        
        return mask_matrix, indices_array, np.array(intensity_val_array)
        
        
    def _swapIndArray(self, matrix):
        """
        Swap the matrix's rows. Limited for two-row matrix only. 
        
        Parameters:
        ----------
            matrix : 2D Float array. 
                The input matrix.

        Returns:
        -------
            indices_array_new: 2D Float matrix. 
                The new matrix with its rows swapped. 
        """
        
        indices_array_new = np.zeros(shape=matrix.shape)
        indices_array_new[0,:] = matrix[1,:]
        indices_array_new[1,:] = matrix[0,:]
        
        return indices_array_new
    
    
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
        
        intensity_thrsld_list = []
        intensity_val_array_sorted = self._getIntensityDistribution(self._intensity_range)
        
        for item in self._pixel_percentage_thrsld:
            index_temp = int(np.floor(item*len(intensity_val_array_sorted))) - 1
            if index_temp < 0: index_temp = 0
            intensity_thrsld_list.append(intensity_val_array_sorted[index_temp])
        
        self._indicesExtraction(intensity_thrsld_list) # Filtered based on percentage. 
        # self._indicesExtraction(self._pixel_intensity_thrsld) # Filtered based on intensity. 
        
        for i in range(self.indices_array.shape[1]):
            ki = tuple(self.indices_array[:,i])
            self._cluster_dict[ki] = None


    def _getNeighbors(self, ind_tuple, label):
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
        
        row_min, row_max, col_min, col_max = self._getSearchRanges(ind_tuple, 
                                                                   (0, self._mask_matrix.shape[0]), 
                                                                   (0, self._mask_matrix.shape[1]))
        neighbor_list = []
        
        for i in range(row_min, row_max):
            for j in range(col_min, col_max):
                if self._mask_matrix[i,j] == label: neighbor_list.append((i,j))
                else: continue
        
        return neighbor_list
    
    
    def _getSearchRanges(self, centerPt, row_range, col_range):
        """
        Get the range of neighbor searching.  
        
        Parameters:
        ----------
            centerPt: Int tuple. 
                The coordinate of center point. Format: (row, col).   
            row_range: Int tuple. 
                The search range on row-axis (axis-0). 
            col_range: Int tuple. 
                The search range on column-axis (axis-1). 

        Returns:
        -------
            row_min: Int. 
                The minimal limit of row searching range. 
            row_max: Int. 
                The maximal limit of row searching range. 
            col_min: Int
                The minimal limit of column searching range. 
            col_max: Int
                The maximal limit of column searching range. 
        """
        
        if centerPt[0] - self._epsilon >= row_range[0]: 
            row_min = centerPt[0] - self._epsilon
        else: row_min = row_range[0]
        
        if centerPt[0] + self._epsilon + 1 <= row_range[1]:
            row_max = centerPt[0] + self._epsilon + 1
        else: row_max = row_range[1]
        
        if centerPt[1] - self._epsilon >= col_range[0]: 
            col_min = centerPt[1] - self._epsilon
        else: col_min = col_range[0]
        
        if centerPt[1] + self._epsilon + 1 <= col_range[1]:
            col_max = centerPt[1] + self._epsilon + 1
        else: col_max = col_range[1]
        
        return row_min, row_max, col_min, col_max
    
    
    # ======================================================================= #  
    
    
    def clustering_old(self):
        """
        """
        
        self._clusterDictEstablishment()
        
        cluster_num = 0
        for ki in self._cluster_dict.keys():
            if self._cluster_dict[ki] != None: continue
            neighbor_list_temp = self._getNeighbors(ki, self._mask_matrix[ki[0], ki[1]])
            
            if len(neighbor_list_temp) < self._minPts: self._cluster_dict[ki] = 0
            else:
                cluster_num += 1
                self._cluster_dict[ki] = cluster_num
                self._clusterExpansion(neighbor_list_temp, cluster_num)
        
        for i in range(cluster_num+1):
            self._cluster_list_dict[i] = copy.deepcopy([key for (key, val) 
                                                        in self._cluster_dict.items() 
                                                        if val == i])
        
        self._clusterNum = cluster_num
    
    
    def _clusterExpansion_old(self, neighbor_list, cluster_num):
        """
        """
        
        for item in neighbor_list:
            if self._cluster_dict[item] == None or self._cluster_dict[item] == 0:
                self._cluster_dict[item] = cluster_num
            else: continue
            
            new_neighbor_list = self._getNeighbors(item, self._mask_matrix[item[0], item[1]])
            
            if len(new_neighbor_list) < self._minPts: continue
            self._clusterExpansion(new_neighbor_list, cluster_num)
        
    
    # ======================================================================= #
    
    
    def clustering(self):
        """
        """
        
        self._clusterDictEstablishment()
        
        cluster_num = 0
        for ki in self._cluster_dict.keys():
            if self._cluster_dict[ki] != None: continue
            neighbor_list_temp = self._getNeighbors(ki, self._mask_matrix[ki[0], ki[1]])
            
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
            
            new_neighbor_list = self._getNeighbors(item, self._mask_matrix[item[0], item[1]])
            
            if len(new_neighbor_list) < self._minPts: continue
            else: new_cluster_pts_list += list(set(new_neighbor_list).difference(cluster_pts_list))
        
        cluster_pts_list = list(set(cluster_pts_list))
        new_cluster_pts_list = list(set(new_cluster_pts_list))
        
        if len(cluster_pts_list) > cluster_size: 
            return self._clusterExpansion(cluster_pts_list, new_cluster_pts_list)
        else: return cluster_pts_list


    def _getIntensityDistribution(self, intensity_range):
        """
        """
        
        _, _, intensity_val_array = self._indicesExtraction_IO(self.data_matrix, 
                                                               intensity_range)
        
        order = np.argsort(intensity_val_array)
        intensity_val_array_sorted = intensity_val_array[order]
        
        return intensity_val_array_sorted    
    
    
    def getBigMostCluster(self):
        """
        """
        
        ki_biggest, length = 0, 0 # Default largest cluster: noise. 
        for (ki, val) in self._cluster_list_dict.items():
            if ki == 0: continue
            if len(val) > length: 
                ki_biggest = ki
                length = len(val)
        
        cluster_biggest = np.transpose(np.array(self._cluster_list_dict[ki_biggest]).astype(int))
        
        return self._swapIndArray(cluster_biggest)
    
    
    def extractClusterIntensities(self, cluster_indices_array):
        """
        """
        
        cluster_val_list = []
        for i in range(cluster_indices_array.shape[1]):
            index_temp = cluster_indices_array[:,i]
            cluster_val_list.append(self.data_matrix[int(index_temp[0]), 
                                                     int(index_temp[1])])
        
        return np.array(cluster_val_list).reshape(-1,)
        
    
    def clusterShapeModeling(self, cluster_indices_array, eigVect_1, eigVect_2, mean_vect):
        """
        """
        
        pos_x_1_ind, pos_x_2_ind, neg_x_1_ind, neg_x_2_ind = 0, 0, 0, 0
        pos_x_1_val, pos_x_2_val, neg_x_1_val, neg_x_2_val = -1e5, -1e5, 1e5, 1e5
        
        for i in range(cluster_indices_array.shape[1]):
            target_vector = cluster_indices_array[:,i].reshape(-1,) - mean_vect.reshape(-1,)
            projection_1_temp = np.dot(target_vector, eigVect_1)
            projection_2_temp = np.dot(target_vector, eigVect_2)
            
            if projection_1_temp >= 0 and projection_1_temp > pos_x_1_val: 
                pos_x_1_val = projection_1_temp
                pos_x_1_ind = i
                continue
                
            if projection_1_temp <= 0 and projection_1_temp < neg_x_1_val: 
                neg_x_1_val = projection_1_temp
                neg_x_1_ind = i
                continue
            
            if projection_2_temp >= 0 and projection_2_temp > pos_x_2_val: 
                pos_x_2_val = projection_2_temp
                pos_x_2_ind = i
                continue
                
            if projection_2_temp <= 0 and projection_2_temp < neg_x_2_val: 
                neg_x_2_val = projection_2_temp
                neg_x_2_ind = i
                continue
                
        anchor_pts_list = [tuple(cluster_indices_array[:,pos_x_1_ind]),
                           tuple(cluster_indices_array[:,neg_x_1_ind]),
                           tuple(cluster_indices_array[:,pos_x_2_ind]),
                           tuple(cluster_indices_array[:,neg_x_2_ind])]
        
        return anchor_pts_list
            
    
    def _clusterVisualization_SINGLE(self, cluster_indices_array):
        """
        """
        
        plt.figure(figsize=(20,20))
        plt.rcParams.update({"font.size": 35})
        plt.tick_params(labelsize=35)
        plt.imshow(self.data_matrix, cmap='gray')
        
        for i in range(cluster_indices_array.shape[1]):
            plt.scatter(cluster_indices_array[0,i], cluster_indices_array[1,i], 
                        c='g', marker='.', linewidths=1.0)
    
    
    def intensityDistributionPlot(self, img_name, n_bins=20, percentage=0.9):
        """
        """
        
        intensity_val_array_sorted = self._getIntensityDistribution(self._intensity_range)
        index_temp = int(np.floor(percentage*len(intensity_val_array_sorted))) - 1
        plot_array = intensity_val_array_sorted[index_temp:]
        
        fig, axes = plt.subplots(1,1,figsize = (20, 12.8), 
                                 sharey=True, tight_layout=True)
        plt.rcParams.update({"font.size": 35})
        plt.tick_params(labelsize=35)
        axes.hist(plot_array, bins=n_bins)
        plt.xlabel("Intensity values", fontsize=40)
        plt.ylabel("Freq", fontsize=40)
        plt.savefig(img_name)
        
    
    def intensityDistribution_projected_Plot(self, img_name, indices_array, mean_vect, 
                                             eigVect_1, eigVect_2, n_bins=20):
        """
        """
        proj_1_list, proj_2_list = [], []
        
        for i in range(indices_array.shape[1]):
            target_vector_temp = indices_array[:,i].reshape(-1,) - mean_vect.reshape(-1,)
            proj_1_temp = np.dot(eigVect_1, target_vector_temp)
            proj_2_temp = np.dot(eigVect_2, target_vector_temp)
            
            proj_1_list.append(proj_1_temp)
            proj_2_list.append(proj_2_temp)
        
        proj_1_max, proj_1_min = max(proj_1_list), min(proj_1_list)
        proj_2_max, proj_2_min = max(proj_2_list), min(proj_2_list)
        
        proj_1_array = 2 * (np.array(proj_1_list) - proj_1_min) / abs(proj_1_max - proj_1_min) - 1
        proj_2_array = 2 * (np.array(proj_2_list) - proj_2_min) / abs(proj_2_max - proj_2_min) - 1
        
        fig, axes = plt.subplots(2, figsize = (20, 12.8), sharey=True, tight_layout=True)
        plt.rcParams.update({"font.size": 35})
        plt.tick_params(labelsize=35)
        
        axes[0].hist(proj_1_array, bins=n_bins)        
        axes[1].hist(proj_2_array, bins=n_bins)
        
        plt.setp(axes[0], xlabel="Axis-1 projection")
        plt.setp(axes[0], ylabel="Freq")
        plt.setp(axes[1], xlabel="Axis-2 projection")
        plt.setp(axes[1], ylabel="Freq")
        
        plt.savefig(img_name)

    
    def clusterVisualization(self, img_name):
        """
        """
        
        plt.figure(figsize=(20,20))
        plt.rcParams.update({"font.size": 35})
        plt.tick_params(labelsize=35)
        plt.imshow(self.data_matrix, cmap='gray')
        
        for ki in self._cluster_dict.keys():
            if self._cluster_dict[ki] == 0: continue # Don't visualize noise. 
            plt.scatter(ki[1], ki[0], 
                        c=self._colorList[self._cluster_dict[ki] % len(self._colorList)], 
                        marker=self._markerList[self._cluster_dict[ki] % len(self._markerList)], 
                        linewidths=1.0)
        
        plt.savefig(img_name)
        
    
    def meltPoolVisualization(self, cluster_indices_array, anchor_pts_list, img_name):
        """
        """
        
        self._clusterVisualization_SINGLE(cluster_indices_array)
            
        for pt in anchor_pts_list:
            plt.scatter(pt[0], pt[1], c='r', marker='*', linewidths=10.0)
        
        plt.savefig(img_name)


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
    

def computeImageMoment(indices_array, val_array, eigVect_1, eigVect_2, mean_vect):
    """
    """
    
    val_array = val_array / np.mean(val_array)

    image_moment, image_moment_unintensified = 0, 0
    for i in range(indices_array.shape[1]):
        val_temp = val_array[i]
        target_vector_temp = indices_array[:,i].reshape(-1,) - mean_vect.reshape(-1,)
        increm_1_sqr = np.dot(eigVect_1, target_vector_temp)**2
        increm_2_sqr = np.dot(eigVect_2, target_vector_temp)**2
        
        image_moment_increm = increm_1_sqr * increm_2_sqr * val_temp
        image_moment_increm_unintensified = increm_1_sqr * increm_2_sqr
        
        image_moment += image_moment_increm
        image_moment_unintensified += image_moment_increm_unintensified
    
    return image_moment / indices_array.shape[1]**2, image_moment_unintensified / indices_array.shape[1]**2


def resultPlot_PCA(img, vector_1, vector_2, mean_vect, img_name):
    """
    """
    
    plt.figure(figsize=(20,20))
    plt.rcParams.update({"font.size": 35})
    plt.tick_params(labelsize=35)
    plt.imshow(img, cmap='gray')
    plt.scatter(mean_vect[0], mean_vect[1], c='r', linewidths=20.0)
    plt.quiver(mean_vect[0], mean_vect[1], vector_1[1], vector_1[0], scale=12.0, color='cyan')
    plt.quiver(mean_vect[0], mean_vect[1], vector_2[1], vector_2[0], scale=12.0, color='cyan')
    plt.savefig(img_name)
    

if __name__ == "__main__":
    input_directory, result_directory = "images", "result"
    image_path_list = os.listdir(input_directory)
    
    PC_num = 2 # Number of principal components. 
    n_bins = 20 # Number of bins for intensity histogram plot.
    
    for image in image_path_list:
        img_name = image.split('.')[0]
        if not os.path.isdir(os.path.join(result_directory, img_name)):
            os.mkdir(os.path.join(result_directory, img_name))
        
        file_path = "{}.png".format(img_name) # The file path of the target image. 
        PCA_plot_name = "r{}_PCA.png".format(img_name)
        cluster_plot_name = "r{}_cluster.png".format(img_name)
        melt_pool_plot_name = "r{}_meltPool.png".format(img_name)
        intensity_plot_name = "r{}_intensity.png".format(img_name)
        intensity_projected_plot_name = "r{}_intensity_projected.png".format(img_name)
        
        img = mig.imread(os.path.join(input_directory, file_path))
        
        dbscan = DBSCAN(img)
        dbscan.clustering() # Implement DBSCAN clustering on the target image. 
        
        melt_pool_indices_array = dbscan.getBigMostCluster() # Extract pixels of largest cluster - melt pool. Swapped. 
        melt_pool_val_array = dbscan.extractClusterIntensities(dbscan._swapIndArray(melt_pool_indices_array)) # Extract corresponding pixel values of largest cluster - melt pool. 
        eigVect, eigVal, _, mean_vect = PCA(melt_pool_indices_array, PC_num) # Implement PCA on melt pool cluster only. 
        
        image_moment, image_moment_unintensified = computeImageMoment(melt_pool_indices_array, 
                                                                      melt_pool_val_array, 
                                                                      eigVect[:,0], eigVect[:,1], 
                                                                      mean_vect) # Compute the image moment of two main axes of the melt pool. 
        
        anchor_pts_list = dbscan.clusterShapeModeling(melt_pool_indices_array, eigVect[:,0], 
                                                      eigVect[:,1], mean_vect)
        dbscan.meltPoolVisualization(melt_pool_indices_array, anchor_pts_list, 
                                     os.path.join(result_directory, img_name, melt_pool_plot_name))
        
        resultPlot_PCA(img, eigVect[:,0], eigVect[:,1], mean_vect, 
                       os.path.join(result_directory, img_name, PCA_plot_name))
        dbscan.clusterVisualization(os.path.join(result_directory, img_name, 
                                                 cluster_plot_name))
        dbscan.intensityDistributionPlot(os.path.join(result_directory, img_name, intensity_plot_name), n_bins)
        dbscan.intensityDistribution_projected_Plot(os.path.join(result_directory, img_name, intensity_projected_plot_name), 
                                                    melt_pool_indices_array, mean_vect, 
                                                    eigVect[:,0], eigVect[:,1], n_bins)
        
        print("{} | IM: {} | IM_uni: {}".format(image, image_moment, image_moment_unintensified))

