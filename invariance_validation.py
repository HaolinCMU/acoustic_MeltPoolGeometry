# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:31:09 2021

@author: hlinl
"""

import os
import ImageMomentInvariants as IMI
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


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
        # if i != 6: continue
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
        # if i != 3: continue
        plt.plot(theta_range, valid_image_moment_PCA_plot_array[:,i], label="PCA_{}".format(i+1), 
                 color=colors_list[i%len(colors_list)], marker=markers_list[i%len(markers_list)], 
                 markersize=5.0, linewidth=5.0)
    
    plt.xlabel("Rotation angle (theta) / degree", fontsize=40)
    plt.ylabel("Image moments - PCA", fontsize=40)
    plt.legend(loc="upper right", prop={"size": 40})
    plt.savefig(file_name)


if __name__ == "__main__":
    file_directory = "result_invariants_archive"
    file_paths_list = os.listdir(file_directory)
    
    file = file_paths_list[1234]
    mdict = scipy.io.loadmat(os.path.join(file_directory, file))
    image_matrix = mdict["image_matrix"]
    indices_array = mdict["indices_array"]
    mean_vect = mdict["mean_vect"].reshape(-1,1)
    eigVect_1, eigVect_2 = mdict["eigVect"][:,0], mdict["eigVect"][:,1]
    
    theta_range, valid_Hu_invariants_array = IMI.valid_Hu_invariance_rotation(image_matrix, indices_array)
    valid_Hu_invariance_rotation_Plot(theta_range, valid_Hu_invariants_array, 
                                          file_name="valid_Hu_{}.png".format(file.split('.')[0]))
    
    theta_range, valid_PCA_invariants_array = IMI.valid_PCA_invariance_rotation(image_matrix, 
                                                                                indices_array, 
                                                                                mean_vect, 
                                                                                eigVect_1, eigVect_2)
    valid_PCA_invariance_rotation_Plot(theta_range, valid_PCA_invariants_array, 
                                           file_name="valid_PCA_{}.png".format(file.split('.')[0]))
    
    