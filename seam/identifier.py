import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.patches as patches
from .utils import plot_pairwise_matrix  # We'll move the plotting function here

class Identifier:
    """Class for identifying and analyzing patterns in MSM data."""
    
    def __init__(self, msm_df, column='Entropy'):
        """
        Initialize Identifier with MSM data.
        
        Parameters
        ----------
        msm_df : pd.DataFrame
            Mechanism Summary Matrix dataframe
        column : str, optional
            Column to analyze (default: 'Entropy')
        """
        self.df = msm_df
        self.column = column
        self.nC = self.df['Cluster'].max() + 1
        self.nP = self.df['Position'].max() + 1
        
        # Create pivot table for analysis
        self.revels = self.df.pivot(
            columns='Position', 
            index='Cluster', 
            values=self.column
        )
        
        # Calculate covariance matrix
        self.cov_matrix = self.revels.cov()
        
    def plot_covariance(self, view_window=None, save_dir=None):
        """
        Plot the covariance matrix.
        
        Parameters
        ----------
        view_window : tuple, optional
            (start, end) positions to view
        save_dir : str, optional
            Directory to save the plot
        """
        matrix = self.cov_matrix.to_numpy()
        
        if view_window:
            matrix = matrix[view_window[0]:view_window[1], 
                          view_window[0]:view_window[1]]
            
        matrix = matrix.reshape(matrix.shape[0], 1, matrix.shape[0], 1)
        
        fig = plot_pairwise_matrix(
            matrix, 
            view_window=view_window, 
            alphabet=['A','C','G','T'],
            cbar_title='Covariance',
            gridlines=False,
            save_dir=save_dir
        )
        
        return fig


