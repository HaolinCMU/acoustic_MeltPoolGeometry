# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 21:58:57 2021

@author: hlinl
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def swapIndArray(matrix):
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


def computeImageMoment_PCA(indices_array, val_array, eigVect_1, eigVect_2, mean_vect):
    """
    indices_array swapped. (spatial_coord_array)
    """
    
    # val_array = val_array / np.mean(val_array)

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
    
    return [image_moment, image_moment_unintensified, 
            image_moment / val_array.shape[0]**2, 
            image_moment_unintensified / val_array.shape[0]**2]


def extractIntensityValArray(image_matrix, indices_array):
    """
    indices_array unswapped. 
    """
    
    val_list = []
    
    for i in range(indices_array.shape[1]):
        r_temp, c_temp = indices_array[0,i], indices_array[1,i]
        val_temp = image_matrix[r_temp, c_temp]
        val_list.append(val_temp)
    
    return np.array(val_list)


def compute_M(p, q, image_matrix, indices_array, spatial_coord_array):
    """
    Raw image moments. 
    indices_array unswapped. 
    """
    
    M_pq = 0
    # spatial_coord_array = swapIndArray(indices_array)
    
    for k in range(indices_array.shape[1]):
        i_temp, j_temp = indices_array[:,k][0], indices_array[:,k][1]
        x_temp, y_temp = spatial_coord_array[:,k][0], spatial_coord_array[:,k][1]
        I_temp = image_matrix[i_temp, j_temp]
        
        M_pq += x_temp**p * y_temp**q * I_temp
    
    return M_pq


def compute_Mu(p, q, image_matrix, indices_array, spatial_coord_array, 
               centerPt=np.array([None, None])):
    """
    indices_array unswapped. 
    """
    
    Mu_pq = 0
    # spatial_coord_array = swapIndArray(indices_array)
    
    if centerPt.any() == None: # Use raw moment to compute central moment. 
        M_00 = compute_M(0, 0, image_matrix, indices_array, spatial_coord_array)
        M_10 = compute_M(1, 0, image_matrix, indices_array, spatial_coord_array)
        M_01 = compute_M(0, 1, image_matrix, indices_array, spatial_coord_array)
        
        x_bar, y_bar = M_10/M_00, M_01/M_00
        
        for k in range(indices_array.shape[1]):
            i_temp, j_temp = indices_array[:,k][0], indices_array[:,k][1]
            x_temp, y_temp = spatial_coord_array[:,k][0], spatial_coord_array[:,k][1]
            I_temp = image_matrix[i_temp, j_temp]
            
            Mu_pq += (x_temp-x_bar)**p * (y_temp-y_bar)**q * I_temp
            
    else: # Use PCA mean-shifting result to compute central moment. 
        x_bar, y_bar = centerPt[0], centerPt[1]
        
        for k in range(indices_array.shape[1]):
            i_temp, j_temp = indices_array[:,k][0], indices_array[:,k][1]
            x_temp, y_temp = spatial_coord_array[:,k][0], spatial_coord_array[:,k][1]
            I_temp = image_matrix[i_temp, j_temp]
            
            Mu_pq += (x_temp-x_bar)**p * (y_temp-y_bar)**q * I_temp
    
    return Mu_pq


def compute_Eta(p, q, image_matrix, indices_array, spatial_coord_array, 
                centerPt=np.array([None, None])):
    """
    indices_array unswapped.
    """
    
    Mu_00 = compute_Mu(0, 0, image_matrix, indices_array, spatial_coord_array, 
                       centerPt=centerPt)
    Mu_pq = compute_Mu(p, q, image_matrix, indices_array, spatial_coord_array, 
                       centerPt=centerPt)
    
    # Eta_pq = Mu_pq / (Mu_00**((p+q)/2) + 1)
    Eta_pq = Mu_pq # Make it scale-dependent. 
    
    return Eta_pq


def compute_Hu_Invariants(image_matrix, indices_array, spatial_coord_array, 
                          centerPt=np.array([None, None])):
    """
    indices_array unswapped.
    """
    
    Eta_11 = compute_Eta(1, 1, image_matrix, indices_array, spatial_coord_array, 
                         centerPt=centerPt)
    Eta_20 = compute_Eta(2, 0, image_matrix, indices_array, spatial_coord_array, 
                         centerPt=centerPt)
    Eta_02 = compute_Eta(0, 2, image_matrix, indices_array, spatial_coord_array, 
                         centerPt=centerPt)
    Eta_12 = compute_Eta(1, 2, image_matrix, indices_array, spatial_coord_array, 
                         centerPt=centerPt)
    Eta_21 = compute_Eta(2, 1, image_matrix, indices_array, spatial_coord_array, 
                         centerPt=centerPt)
    Eta_30 = compute_Eta(3, 0, image_matrix, indices_array, spatial_coord_array, 
                         centerPt=centerPt)
    Eta_03 = compute_Eta(0, 3, image_matrix, indices_array, spatial_coord_array, 
                         centerPt=centerPt)
    
    I_1 = Eta_20 + Eta_02
    I_2 = (Eta_20 - Eta_02)**2 + 4*Eta_11**2
    I_3 = (Eta_30 - 3*Eta_12)**2 + (3*Eta_21 - Eta_03)**2
    I_4 = (Eta_30 + Eta_12)**2 + (Eta_21 + Eta_03)**2
    I_5 = ((Eta_30 - 3*Eta_12)*(Eta_30 + Eta_12)*((Eta_30 + Eta_12)**2 - 3*(Eta_21 + Eta_03)**2) + 
           (3*Eta_21 - Eta_03)*(Eta_21 + Eta_03)*(3*(Eta_30 + Eta_12)**2 - (Eta_21 + Eta_03)**2))
    I_6 = ((Eta_20 - Eta_02)*((Eta_30 + Eta_12)**2 - (Eta_21 + Eta_03)**2) + 
           4*Eta_11*(Eta_30 + Eta_12)*(Eta_21 + Eta_03))
    I_7 = ((3*Eta_21 - Eta_03)*(Eta_30 + Eta_12)*((Eta_30 + Eta_12)**2 - 3*(Eta_21 + Eta_03)**2) - 
           (Eta_30 - 3*Eta_12)*(Eta_21 + Eta_03)*(3*(Eta_30 + Eta_12)**2 - (Eta_21 + Eta_03)**2))
    
    # return [I_1, I_2, I_3, I_4, I_5, I_6, I_7]
    
    # ===================== Use off-the-shelf cv2 package ======================== #
    
    # No customization for central point is permitted. 
    image_matrix_filtered = np.zeros(shape=image_matrix.shape)
    
    for k in range(indices_array.shape[1]):
        i_temp, j_temp = indices_array[0,k], indices_array[1,k]
        image_matrix_filtered[i_temp, j_temp] = image_matrix[i_temp, j_temp]
    
    moments = cv2.moments(image_matrix_filtered)
    Hu_moments_array = cv2.HuMoments(moments)
    # Hu_moments_array = np.log(np.abs(Hu_moments_array))
    
    return list(list(Hu_moments_array.reshape(1,-1))[0])

    # ============================================================================ #


def computeSkewness(image_matrix, indices_array):
    """
    """
    
    
    
    pass


def computeKurtosis():
    """
    """
    
    pass


def coordRotate(spatial_coord_array, theta):
    """
    2 x N. 
    """
    
    theta_rad = theta * np.pi / 180.0
    rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                                [np.sin(theta_rad), np.cos(theta_rad)]])
    
    spatial_coord_array_rotated = np.dot(rotation_matrix, spatial_coord_array)
    
    return spatial_coord_array_rotated


def valid_Hu_invariance_rotation(image_matrix, indices_array, centerPt=np.array([None, None])):
    """
    """
    
    theta_range = np.linspace(-90.0, 90.0, 19)
    
    Hu_invariants_array_list = []
    spatial_coord_array = swapIndArray(indices_array)
    
    for theta in theta_range:
        spatial_coord_array_rotated_temp = coordRotate(spatial_coord_array, theta)
        if centerPt.all() != None: centerPt_rotated = coordRotate(centerPt, theta)
        else: centerPt_rotated = centerPt
        
        Hu_invariants_list_temp = compute_Hu_Invariants(image_matrix, indices_array, 
                                                        spatial_coord_array_rotated_temp, 
                                                        centerPt=centerPt_rotated)
        Hu_invariants_array_list.append(Hu_invariants_list_temp)
    
    valid_Hu_invariants_array = np.array(Hu_invariants_array_list)
    
    return theta_range, valid_Hu_invariants_array


def valid_PCA_invariance_rotation(image_matrix, indices_array, mean_vect, eigVect_1, eigVect_2):
    """
    """
    
    theta_range = np.linspace(-90.0, 90.0, 19)
    
    image_moment_PCA_list = []
    spatial_coord_array = swapIndArray(indices_array)
    val_array = extractIntensityValArray(image_matrix, indices_array)
    
    for theta in theta_range:
        spatial_coord_array_rotated_temp = coordRotate(spatial_coord_array, theta)
        mean_vect_rotated_temp = coordRotate(mean_vect, theta)
        eigVect_1_rotated_temp = coordRotate(eigVect_1, theta)
        eigVect_2_rotated_temp = coordRotate(eigVect_2, theta)
        
        PCA_invariants_list = computeImageMoment_PCA(spatial_coord_array_rotated_temp, 
                                                     val_array, eigVect_1_rotated_temp, 
                                                     eigVect_2_rotated_temp, mean_vect_rotated_temp)
        
        image_moment_PCA_list.append(PCA_invariants_list)
    
    valid_image_moment_PCA_plot_array = np.array(image_moment_PCA_list) # col (from left to right): image_moment, image_moment_unintensified, image_moment_normalized, image_moment_normalized_unintensified
    
    return theta_range, valid_image_moment_PCA_plot_array


def valid_Hu_invariance_rotation_Plot(theta_range, valid_Hu_invariants_array,
                                      file_name):
    """
    """
    
    colors_list = ['red', 'green', 'blue', 'cyan', 'purple', 'yellow', 'brown']
    markers_list = ['.', '*', '^', 'o', 'v', '+', 'x']
    
    plt.figure(figsize=(20,20))
    plt.rcParams.update({"font.size": 35})
    plt.tick_params(labelsize=35)
    
    for i in range(valid_Hu_invariants_array.shape[1]):
        plt.plot(theta_range, valid_Hu_invariants_array[:,i], label="I_{}".format(i+1), 
                 color=colors_list[i%len(colors_list)], marker=markers_list[i%len(markers_list)], 
                 markersize=5.0, linewidth=5.0)
    
    plt.xlabel("Rotation angle (theta) / degree", fontsize=40)
    plt.ylabel("Hu invariants", fontsize=40)
    plt.legend(loc="upper right", prop={"size": 40})
    plt.savefig(file_name)
    
    
def valid_PCA_invariance_rotation_Plot(theta_range, valid_image_moment_PCA_plot_array,
                                       file_name):
    """
    """
    
    colors_list = ['red', 'green', 'blue', 'cyan', 'purple', 'yellow', 'brown']
    markers_list = ['.', '*', '^', 'o', 'v', '+', 'x']
    
    plt.figure(figsize=(20,20))
    plt.rcParams.update({"font.size": 35})
    plt.tick_params(labelsize=35)
    
    for i in range(valid_image_moment_PCA_plot_array.shape[1]):
        plt.plot(theta_range, valid_image_moment_PCA_plot_array[:,i], label="var_{}".format(i+1), 
                 color=colors_list[i%len(colors_list)], marker=markers_list[i%len(markers_list)], 
                 markersize=5.0, linewidth=5.0)
    
    plt.xlabel("Rotation angle (theta) / degree", fontsize=40)
    plt.ylabel("Image moments - PCA", fontsize=40)
    plt.legend(loc="upper right", prop={"size": 40})
    plt.savefig(file_name)
    
