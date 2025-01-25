from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from Logo import Logo
from gpu_utils import GPUTransformer
from colors import get_rgb, get_color_dict, CHARS_TO_COLORS_DICT, COLOR_SCHEME_DICT
from tqdm.auto import tqdm
from Glyph import Glyph
import matplotlib.font_manager as fm

class TimingContext:
    def __init__(self, name, timing_dict):
        self.name = name
        self.timing_dict = timing_dict
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.timing_dict[self.name] = time.time() - self.start

class BatchLogoGPU:
    def __init__(self, attribution_array, alphabet, **kwargs):
        """Initialize BatchLogoGPU with attribution array and parameters"""
        self.attribution_array = attribution_array
        self.N = attribution_array.shape[0]  # number of logos
        self.L = attribution_array.shape[1]  # length of each logo
        self.alphabet = alphabet
        self.kwargs = self._get_default_kwargs()
        self.kwargs.update(kwargs)
        
        # Initialize storage for processed logos
        self.processed_logos = {}
        
        # Set batch size
        self.batch_size = self.kwargs.pop('batch_size', 10)
        
        # Set figure size
        self.figsize = self.kwargs.pop('figsize', (10, 2.5))
        
        # Get font name
        self.font_name = self.kwargs.pop('font_name', 'sans')
        
        # Get stack order
        self.stack_order = self.kwargs.pop('stack_order', 'big_on_top')
        
        # Get color scheme (default to 'classic' for DNA/RNA)
        color_scheme = self.kwargs.pop('color_scheme', 'classic')
        
        # Get RGB colors using same method as Logo class
        self.rgb_dict = get_color_dict(color_scheme, self.alphabet)

    def _get_font_props(self):
        """Get cached font properties"""
        if self.font_name not in self._font_cache:
            self._font_cache[self.font_name] = fm.FontProperties(family=self.font_name)
        return self._font_cache[self.font_name]

    def process_all(self):
        """Process all logos in batches"""
        with tqdm(total=self.N, desc="Processing logos") as pbar:
            for start_idx in range(0, self.N, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.N)
                self._process_batch(start_idx, end_idx)
                pbar.update(end_idx - start_idx)
        return self
    
    def _process_batch(self, start_idx, end_idx):
        """Process a batch of logos and store their data"""
        for idx in range(start_idx, end_idx):
            fig, ax = plt.subplots(figsize=self.figsize)
            glyph_list = []
            
            for pos in range(self.L):
                values = self.attribution_array[idx, pos]
                
                if self.stack_order == 'big_on_top':
                    ordered_indices = np.argsort(values)
                elif self.stack_order == 'small_on_top':
                    tmp_vs = np.zeros(len(values))
                    indices = (values != 0)
                    tmp_vs[indices] = 1.0/values[indices]
                    ordered_indices = np.argsort(tmp_vs)
                else:  # fixed
                    ordered_indices = np.array(range(len(values)))[::-1]
                
                values = values[ordered_indices]
                chars = [str(self.alphabet[i]) for i in ordered_indices]
                floor = sum((values - self.kwargs['vsep']) * (values < 0)) + self.kwargs['vsep']/2.0
                
                for value, char in zip(values, chars):
                    if value != 0:
                        ceiling = floor + abs(value)
                        glyph = Glyph(pos, char,
                                    ax=ax,
                                    floor=floor,
                                    ceiling=ceiling,
                                    color=self.rgb_dict[char],
                                    flip=(value < 0 and self.kwargs['flip_below']),
                                    font_name=self.font_name,
                                    alpha=self.kwargs['alpha'],
                                    vpad=self.kwargs['vpad'],
                                    width=self.kwargs['width'],
                                    zorder=0)
                        glyph._make_patch()
                        glyph_list.append(glyph)
                        floor = ceiling + self.kwargs['vsep']
                
            self.processed_logos[idx] = {
                'glyphs': glyph_list,
                'fig': fig,
                'ax': ax
            }
            plt.close(fig)
    
    def draw_logos(self, indices=None, rows=None, cols=None):
        """
        Draw specific logos in a grid layout
        
        Parameters
        ----------
        indices : list or None
            Indices of logos to draw. If None, draws all logos
        rows, cols : int or None
            Grid dimensions. If None, will be automatically determined
        """
        if indices is None:
            indices = list(range(self.N))
        
        N = len(indices)
        
        # Determine grid layout
        if rows is None and cols is None:
            cols = min(5, N)
            rows = (N + cols - 1) // cols
        elif rows is None:
            rows = (N + cols - 1) // cols
        elif cols is None:
            cols = (N + rows - 1) // rows
            
        # Create figure with subplots
        fig, axes = plt.subplots(rows, cols, 
                                figsize=(self.figsize[0]*cols, self.figsize[1]*rows),
                                squeeze=False)
        
        # Draw requested logos
        for i, idx in enumerate(indices):
            if idx not in self.processed_logos:
                raise ValueError(f"Logo {idx} has not been processed yet. Run process_all() first.")
                
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            logo_data = self.processed_logos[idx]
            self._draw_single_logo(ax, logo_data)
            
        # Turn off empty subplots
        for i in range(N, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
            
        plt.tight_layout()
        return fig, axes
    
    def draw_single(self, idx):
        """Draw a single logo"""
        if idx not in self.processed_logos:
            raise ValueError(f"Logo {idx} has not been processed yet. Run process_all() first.")
            
        fig, ax = plt.subplots(figsize=self.figsize)
        self._draw_single_logo(ax, self.processed_logos[idx])
        plt.tight_layout()
        return fig, ax
    
    def _draw_single_logo(self, ax, logo_data):
        """Draw a single logo on the given axes"""
        patches = []
        for glyph in logo_data['glyphs']:
            path = glyph._get_transformed_path()
            patch = PathPatch(path,
                            facecolor=glyph.color,
                            edgecolor=glyph.edgecolor,
                            linewidth=glyph.edgewidth,
                            alpha=glyph.alpha)
            patches.append(patch)
        
        # Add all patches at once
        ax.add_collection(PatchCollection(patches, match_original=True))
        
        # Set proper axis limits
        ax.set_xlim(-0.5, self.L - 0.5)
        
        # Calculate ylims from glyphs
        floors = [g.floor for g in logo_data['glyphs']]
        ceilings = [g.ceiling for g in logo_data['glyphs']]
        ymin = min(floors) if floors else 0
        ymax = max(ceilings) if ceilings else 1
        
        # Ensure baseline is visible
        ymin = min(ymin, 0)
        ax.set_ylim(ymin, ymax)
        
        # Draw baseline
        if self.kwargs['baseline_width'] > 0:
            ax.axhline(y=0, color='black',
                      linewidth=self.kwargs['baseline_width'],
                      zorder=-1)
        
        # Show spines
        if self.kwargs['show_spines']:
            for spine in ax.spines.values():
                spine.set_visible(True)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)

    def _get_default_kwargs(self):
        """Get default parameters for logo creation"""
        return {
            'show_spines': True,
            'baseline_width': 0.5,
            'vsep': 0.0,
            'alpha': 1.0,
            'vpad': 0.0,
            'width': 1.0,
            'flip_below': True,
            'color_scheme': {},
        } 