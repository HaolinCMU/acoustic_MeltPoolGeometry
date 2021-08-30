# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:01:18 2021

@author: hlinl
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import ImageMomentInvariants as IMI
from melt_pool_image_computation import load_results


def sequentialResultsPlot(plot_array, save_fig_name):
    """
    imgNum x 7. 
    """
    
    plot_x = np.arange(plot_array.shape[0])
    
    colors_list = ['red', 'green', 'blue', 'cyan', 'purple', 'yellow', 'brown']
    markers_list = ['.', '*', '^', 'o', 'v', '+', 'x']
    
    plt.figure(figsize=(20,20))
    plt.rcParams.update({"font.size": 35})
    plt.tick_params(labelsize=35)
    
    for i in range(plot_array.shape[1]):
        plt.plot(plot_x, plot_array[:,i], color=colors_list[i%len(colors_list)],
                 label="var_{}".format(i+1), marker=markers_list[i%len(markers_list)], 
                 markersize=5.0, linewidth=5.0)
    
    plt.xlabel("Time frame No. ", fontsize=40)
    plt.ylabel("{}_Invariants".format(save_fig_name), fontsize=40)
    plt.legend(loc="upper right", prop={"size": 40})
    plt.savefig("sequential_{}.png".format(save_fig_name))
    

if __name__ == "__main__":
    result_plot_directory = "result_plot_target"
    file_path_list = os.listdir(result_plot_directory)
    result_directory = "result_invariants_archive"
    Hu_invariants_sequential_plot_figname = "Hu"
    PCA_sequential_plot_figname = "PCA"
    
    for ind, file in enumerate(file_path_list):
        file_name = file.split('.')[0]
        mdict = load_results(os.path.join(result_directory, "{}.mat".format(file_name)))
        image_Hu_invariants_array = mdict["image_Hu_invariants_array"][:,:]
        PCA_result = mdict["image_PCA_invariants_array"][:,2:]
        
        if ind == 0: 
            Hu_invariants_sequential_plot_array = image_Hu_invariants_array
            PCA_sequential_plot_array = PCA_result
        else: 
            Hu_invariants_sequential_plot_array = np.vstack((Hu_invariants_sequential_plot_array,
                                                             image_Hu_invariants_array))
            PCA_sequential_plot_array = np.vstack((PCA_sequential_plot_array, PCA_result))
        
    sequentialResultsPlot(Hu_invariants_sequential_plot_array, Hu_invariants_sequential_plot_figname)
    sequentialResultsPlot(PCA_sequential_plot_array, PCA_sequential_plot_figname)
    
    
    