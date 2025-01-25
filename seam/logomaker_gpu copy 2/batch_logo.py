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
from colors import get_rgb, get_color_dict
from tqdm.auto import tqdm
from Glyph import Glyph
import matplotlib.font_manager as fm

class BatchLogoGPU:
    def __init__(self, attribution_array, 
                 alphabet=['A', 'C', 'G', 'T'],
                 color_scheme='classic',
                 figsize=(10, 2.5),
                 font_name='sans',
                 stack_order='big_on_top',
                 vsep=0.0,
                 vpad=0.0,
                 width=1.0,
                 baseline_width=0.5,
                 flip_below=True,
                 show_spines=True,
                 alpha=1.0,
                 **kwargs):
        """Initialize BatchLogoGPU."""
        # Validate input array
        if not isinstance(attribution_array, np.ndarray) or attribution_array.ndim != 3:
            raise ValueError("attribution_array must be a 3D numpy array")
        
        self.N, self.L, self.A = attribution_array.shape
        if len(alphabet) != self.A:
            raise ValueError(f"Length of alphabet ({len(alphabet)}) must match "
                           f"last dimension of attribution_array ({self.A})")
        
        self.attribution_array = attribution_array
        self.alphabet = alphabet
        self.color_scheme = color_scheme
        self.figsize = figsize
        self.font_name = font_name
        self.stack_order = stack_order
        
        # Store logo styling parameters
        self.kwargs = {
            'vsep': vsep,
            'vpad': vpad,
            'width': width,
            'baseline_width': baseline_width,
            'flip_below': flip_below,
            'show_spines': show_spines,
            'alpha': alpha,
            'font_name': font_name,
            **kwargs
        }
        
        # Get color dictionary
        self.rgb_dict = get_color_dict(self.color_scheme, self.alphabet)
        
        # Initialize GPU transformer
        try:
            self.gpu_transformer = GPUTransformer()
            print("GPU acceleration enabled")
        except ImportError:
            raise ImportError("GPU acceleration requires TensorFlow")
        
        self.processed_logos = {}
        self.batch_size = kwargs.get('batch_size', 1000)  # Default batch size of 1000
        
        # Initialize font cache
        self._font_cache = {}

    def _get_font_props(self):
        """Get cached font properties"""
        if self.font_name not in self._font_cache:
            self._font_cache[self.font_name] = fm.FontProperties(family=self.font_name)
        return self._font_cache[self.font_name]

    def process_all(self):
        """Process all logos in batches"""
        total_logos = 0
        batch_num = 1
        for start_idx in range(0, self.N, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.N)
            print(f"\nProcessing batch {batch_num} ({end_idx-start_idx} logos)")
            self._process_batch(start_idx, end_idx)
            total_logos += end_idx - start_idx
            batch_num += 1
        print(f"\nProcessed {total_logos} logos in {batch_num-1} batches")
        
        return self
    
    def _process_batch(self, start_idx, end_idx):
        """Process a batch of logos and store their data"""
        total_start = time.time()
        setup_time = 0
        sort_time = 0
        glyph_time = 0
        font_time = 0
        
        # Break down path creation time
        text_path_time = 0
        m_path_time = 0
        
        # Break down transform time
        bbox_calc_time = 0
        stretch_calc_time = 0
        transform_apply_time = 0
        
        for idx in tqdm(range(start_idx, end_idx), 
                       desc=f"Processing logos {start_idx}-{end_idx}",
                       leave=False):
            
            setup_start = time.time()
            fig, ax = plt.subplots(figsize=self.figsize)
            glyph_list = []
            setup_time += time.time() - setup_start
            
            for pos in range(self.L):
                values = self.attribution_array[idx, pos]
                
                sort_start = time.time()
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
                sort_time += time.time() - sort_start
                
                for value, char in zip(values, chars):
                    if value != 0:
                        ceiling = floor + abs(value)
                        
                        # Time font operations
                        font_start = time.time()
                        font_props = self._get_font_props()
                        font_time += time.time() - font_start
                        
                        # Time text path creation
                        text_start = time.time()
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
                        text_path_time += time.time() - text_start
                        
                        # Time M path creation and bbox calculations
                        bbox_start = time.time()
                        bbox_calc = time.time()
                        
                        # Time stretch calculations
                        stretch_start = time.time()
                        glyph._make_patch()  # This will need to be modified to expose timing
                        stretch_calc_time += time.time() - stretch_start
                        
                        # Time final transformation application
                        transform_start = time.time()
                        transform_apply_time += time.time() - transform_start
                        
                        glyph_list.append(glyph)
                        floor = ceiling + self.kwargs['vsep']
                
            self.processed_logos[idx] = {
                'glyphs': glyph_list,
                'fig': fig,
                'ax': ax
            }
            plt.close(fig)
        
        total_time = time.time() - total_start
        print(f"\nTiming breakdown for batch {start_idx//self.batch_size + 1}:")
        print(f"Setup time: {setup_time:.3f}s ({setup_time/total_time*100:.1f}%)")
        print(f"Sort time: {sort_time:.3f}s ({sort_time/total_time*100:.1f}%)")
        print(f"Font operations: {font_time:.3f}s ({font_time/total_time*100:.1f}%)")
        print("\nPath creation details:")
        print(f"  - Text path creation: {text_path_time:.3f}s")
        print(f"  - M path creation: {m_path_time:.3f}s")
        print("\nTransform details:")
        print(f"  - BBox calculations: {bbox_calc_time:.3f}s")
        print(f"  - Stretch calculations: {stretch_calc_time:.3f}s")
        print(f"  - Transform application: {transform_apply_time:.3f}s")
        print(f"\nTotal time: {total_time:.3f}s")
        print(f"Average time per logo: {total_time/(end_idx-start_idx):.3f}s")
    
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