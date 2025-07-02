#!/usr/bin/env python3
"""
Batch Logo Generation Module

This module provides an optimized version of Logomaker with faster logo generation 
and batch processing capabilities.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from logomaker_batch.colors import get_rgb, COLOR_SCHEME_DICT
from tqdm import tqdm
import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.colors import to_rgb


class TimingContext:
    def __init__(self, name, timing_dict):
        self.name = name
        self.timing_dict = timing_dict
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.timing_dict[self.name] = time.time() - self.start

class BatchLogo:
    # Initialize class-level caches
    _path_cache = {}
    _m_path_cache = {}
    _transform_cache = {}
    _font_cache = {}
    
    def __init__(self, values, alphabet=None, figsize=[10,2.5], batch_size=50, gpu=False, font_name='sans', y_min_max=None, ref_seq=None, show_progress=True, **kwargs):
        """Initialize BatchLogo processor"""
        if gpu:
            print("Warning: GPU acceleration not yet implemented, falling back to CPU")
            
        # Reset caches for each new instance
        self._path_cache = {}
        self._m_path_cache = {}
        self._transform_cache = {}
        self._font_cache = {}  # Reset font cache

        # Handle centering if requested
        center_values = kwargs.pop('center_values', False)
        if center_values:
            values = self._center_matrix(values)

        self.values = np.array(values)
        self.alphabet = alphabet if alphabet is not None else ['A', 'C', 'G', 'T']
        self.batch_size = batch_size
        
        self.N = self.values.shape[0]  # number of logos
        self.L = self.values.shape[1]  # length of each logo
        
        self.kwargs = self._get_default_kwargs()
        self.kwargs.update(kwargs)
        
        # Initialize storage for processed logos
        self.processed_logos = {}
        
        # Set figure size
        self.figsize = figsize
        
        # Get font name and weight
        self.font_name = font_name
        self.font_weight = self.kwargs.pop('font_weight', 'normal')
        
        # Get stack order
        self.stack_order = self.kwargs.pop('stack_order', 'big_on_top')
        
        # Get color scheme
        color_scheme = self.kwargs.pop('color_scheme', 'classic')
        
        # Initialize rgb_dict
        self.rgb_dict = {}
        
        # Handle color scheme
        if isinstance(color_scheme, dict):
            # If dictionary provided, use it directly
            for char in self.alphabet:
                self.rgb_dict[char] = get_rgb(color_scheme.get(char, 'gray'))
        else:
            # Otherwise use predefined scheme
            colors = COLOR_SCHEME_DICT[color_scheme]
            for char in self.alphabet:
                if char in colors:
                    self.rgb_dict[char] = get_rgb(colors[char])
                elif char == 'T' and 'TU' in colors:  # Handle T/U case
                    self.rgb_dict[char] = get_rgb(colors['TU'])
                else:
                    self.rgb_dict[char] = get_rgb('gray')

        self.y_min_max = y_min_max

        # Store reference sequence if provided
        self.ref_seq = ref_seq

        self.show_progress = show_progress

    def _get_font_props(self):
        """Get cached font properties with fallback to sans if font not found"""
        cache_key = f"{self.font_name}_{self.font_weight}"
        if cache_key not in self._font_cache:
            try:
                self._font_cache[cache_key] = fm.FontProperties(family=self.font_name, weight=self.font_weight)
            except:
                print(f"Warning: Font '{self.font_name}' with weight '{self.font_weight}' not found, falling back to 'sans'")
                self._font_cache[cache_key] = fm.FontProperties(family='sans')
        return self._font_cache[cache_key]

    def process_all(self):
        """Process all logos in batches"""
        timing = {}
        with TimingContext('total_processing', timing):
            if self.show_progress:
                with tqdm(total=self.N, desc="Processing logos") as pbar:
                    for start_idx in range(0, self.N, self.batch_size):
                        end_idx = min(start_idx + self.batch_size, self.N)
                        with TimingContext(f'batch_{start_idx}_{end_idx}', timing):
                            self._process_batch(start_idx, end_idx)
                        pbar.update(end_idx - start_idx)
            else:
                for start_idx in range(0, self.N, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, self.N)
                    with TimingContext(f'batch_{start_idx}_{end_idx}', timing):
                        self._process_batch(start_idx, end_idx)
        return self
    
    def _process_batch(self, start_idx, end_idx):
        """Process a batch of logos"""
        batch_timing = {}
        
        with TimingContext('batch_total', batch_timing):
            font_props = self._get_font_props()
            
            # Cache M path first (for width reference)
            if not self._m_path_cache:
                m_path = TextPath((0, 0), 'M', size=1, prop=font_props)
                m_extents = m_path.get_extents()
                self._m_path_cache = {
                    'path': m_path,
                    'extents': m_extents,
                    'width': m_extents.width,
                }
                
                # Then cache alphabet paths
                for char in self.alphabet:
                    if char not in self._path_cache:
                        base_path = TextPath((0, 0), char, size=1, prop=font_props)
                        flipped_path = TextPath((0, 0), char, size=1, prop=font_props)
                        flipped_path = flipped_path.transformed(Affine2D().scale(1, -1))
                        self._path_cache[char] = {
                            'normal': {'path': base_path, 'extents': base_path.get_extents()},
                            'flipped': {'path': flipped_path, 'extents': flipped_path.get_extents()}
                        }
            
            for idx in range(start_idx, end_idx):
                with TimingContext(f'logo_{idx}', batch_timing):
                    glyph_data = []
                    
                    for pos in range(self.L):
                        values = self.values[idx, pos]
                        ordered_indices = self._get_ordered_indices(values)
                        values = values[ordered_indices]
                        chars = [str(self.alphabet[i]) for i in ordered_indices]
                        
                        # Calculate total negative height first
                        neg_values = values[values < 0]
                        total_neg_height = abs(sum(neg_values)) + (len(neg_values) - 1) * self.kwargs['vsep']
                        
                        # Handle positive values (stack up from 0)
                        floor = self.kwargs['vsep']/2.0
                        for value, char in zip(values, chars):
                            if value > 0:
                                ceiling = floor + value
                                
                                path_data = self._path_cache[char]['normal']
                                transformed_path = self._get_transformed_path(
                                    path_data, pos, floor, ceiling,
                                    self._m_path_cache['extents'].width
                                )
                                
                                glyph_data.append({
                                    'path': transformed_path,
                                    'color': self.rgb_dict[char],
                                    'edgecolor': 'none',
                                    'edgewidth': 0,
                                    'alpha': self.kwargs['alpha'],
                                    'floor': floor,
                                    'ceiling': ceiling,
                                    'char': char,
                                    'pos': pos
                                })
                                floor = ceiling + self.kwargs['vsep']
                        
                        # Handle negative values (stack down from -total_height)
                        if len(neg_values) > 0:
                            floor = -total_neg_height - self.kwargs['vsep']/2.0
                            for value, char in zip(values, chars):
                                if value < 0:
                                    ceiling = floor + abs(value)
                                    
                                    path_data = self._path_cache[char]['flipped' if self.kwargs['flip_below'] else 'normal']
                                    transformed_path = self._get_transformed_path(
                                        path_data, pos, floor, ceiling,
                                        self._m_path_cache['extents'].width
                                    )
                                    
                                    # Apply fade and shade effects for negative values
                                    alpha = self.kwargs['alpha']
                                    if value < 0:
                                        alpha *= (1 - self.kwargs['fade_below'])
                                        if self.kwargs['shade_below'] > 0:
                                            color = tuple(c * (1 - self.kwargs['shade_below']) for c in self.rgb_dict[char])
                                        else:
                                            color = self.rgb_dict[char]
                                    else:
                                        color = self.rgb_dict[char]
                                    
                                    glyph_data.append({
                                        'path': transformed_path,
                                        'color': color,
                                        'edgecolor': 'none',
                                        'edgewidth': 0,
                                        'alpha': alpha,
                                        'floor': floor,
                                        'ceiling': ceiling,
                                        'char': char,
                                        'pos': pos
                                    })
                                    floor = ceiling + self.kwargs['vsep']
                    
                    self.processed_logos[idx] = {'glyphs': glyph_data}
            
            #print(f"Batch {start_idx}-{end_idx} timing:", batch_timing)
    
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
    
    def draw_single(self, idx, fixed_ylim=True, view_window=None, figsize=None, 
                    highlight_ranges=None, highlight_colors=None, highlight_alpha=0.5,
                    border=True, ax=None):
        """Draw a single logo
        
        Parameters
        ----------
        idx : int
            Index of logo to draw
        fixed_ylim : bool, default=True
            Whether to use same y-axis limits across all logos
        view_window : list or tuple, optional
            [start,end] positions to view. If None, show entire logo
        figsize : tuple, optional
            Figure size in inches. If None, use size from initialization
        highlight_ranges : list of tuple/list, optional
            Either [(start,stop), ...] for continuous ranges
            or [[pos1, pos2, pos3, ...], ...] for specific positions.
            For position lists, contiguous integers are treated as ranges.
        highlight_colors : list of str or str, optional
            Colors for highlighting, e.g. ['lightcyan', 'honeydew'] or 'lightcyan'
            If None, defaults to plt.cm.Pastel1 (9 colors)
        highlight_alpha : float, default=0.5
            Alpha transparency for highlights
        border : bool, default=True
            Whether to show the axis spines (border)
        ax : matplotlib.axes.Axes, optional
            If provided, draw the logo on this axes. If None, create a new figure/axes.
        """
        if idx not in self.processed_logos:
            raise ValueError(f"Logo {idx} has not been processed yet. Run process_all() first.")
        own_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize if figsize is not None else self.figsize)
            own_fig = True
        else:
            fig = None
        self._draw_single_logo(ax, self.processed_logos[idx], fixed_ylim=fixed_ylim, border=border)
        # Add highlighting if specified
        if highlight_ranges is not None:
            # Convert single tuple/list to list of ranges
            if isinstance(highlight_ranges[0], (int, float)):
                highlight_ranges = [highlight_ranges]
            # Set default colors if None
            if highlight_colors is None:
                n_ranges = len(highlight_ranges)
                highlight_colors = [plt.cm.Pastel1(i % 9) for i in range(n_ranges)]
            elif isinstance(highlight_colors, str):
                highlight_colors = [highlight_colors]
            # Add highlighting rectangles (using full sequence coordinates)
            for positions, color in zip(highlight_ranges, highlight_colors):
                # Handle both (start,stop) tuples and [pos1, pos2, ...] lists
                if len(positions) == 2 and isinstance(positions, tuple):
                    # For tuples, highlight the continuous range
                    start, end = positions
                    ax.axvspan(start-0.5, end-0.5, color=color, alpha=highlight_alpha, zorder=-1)
                else:
                    # For position lists, find contiguous runs
                    positions = sorted(positions)
                    start = positions[0]
                    prev = start
                    for curr in positions[1:] + [None]:
                        if curr != prev + 1:
                            # End of a run
                            end = prev
                            if start == end:
                                ax.axvspan(start-0.5, start+0.5, color=color, alpha=highlight_alpha, zorder=-1)
                            else:
                                ax.axvspan(start-0.5, end+0.5, color=color, alpha=highlight_alpha, zorder=-1)
                            start = curr
                        prev = curr
        # Apply view window last
        if view_window is not None:
            start, end = view_window
            ax.set_xlim(start-0.5, end-0.5)
        plt.tight_layout()
        if own_fig:
            return fig, ax
        else:
            return None, ax
    
    def _draw_single_logo(self, ax, logo_data, fixed_ylim=True, border=True):
        """Draw a single logo on the given axes.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on
        logo_data : dict
            Logo data containing glyphs
        fixed_ylim : bool, default=True
            Whether to use same y-axis limits across all logos
        border : bool, default=True
            Whether to show the axis spines (border)
        """
        timing = {}
        
        with TimingContext('patch_creation', timing):
            patches = []
            for glyph_data in logo_data['glyphs']:
                patch = PathPatch(glyph_data['path'],
                                facecolor=glyph_data['color'],
                                edgecolor=glyph_data['edgecolor'],
                                linewidth=glyph_data['edgewidth'],
                                alpha=glyph_data['alpha'])
                patches.append(patch)
        
        with TimingContext('patch_collection', timing):
            ax.add_collection(PatchCollection(patches, match_original=True))
        
        with TimingContext('axis_setup', timing):
            # Set proper axis limits
            ax.set_xlim(-0.5, self.L - 0.5)
            
            if fixed_ylim and self.y_min_max is not None:
                ax.set_ylim(self.y_min_max[0], self.y_min_max[1])
            else:
                # Calculate ylims from glyphs only
                floors = [g['floor'] for g in logo_data['glyphs']]
                ceilings = [g['ceiling'] for g in logo_data['glyphs']]
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
            
            # Show/hide spines based on border parameter
            for spine in ax.spines.values():
                spine.set_visible(border)
        
        #print("Logo drawing details:", timing)

    def _get_default_kwargs(self):
        """Get default parameters for logo creation"""
        return {
            'show_spines': True,
            'baseline_width': 0.5,
            'vsep': 0.0,
            'alpha': 1.0,
            'vpad': 0.0,
            'width': 0.9,
            'flip_below': True,
            'color_scheme': 'classic',
            'fade_below': 0.5,
            'shade_below': 0.5, 
        }

    def _get_ordered_indices(self, values):
        """Get indices ordered according to stack_order"""
        if self.stack_order == 'big_on_top':
            return np.argsort(values)
        elif self.stack_order == 'small_on_top':
            tmp_vs = np.zeros(len(values))
            indices = (values != 0)
            tmp_vs[indices] = 1.0/values[indices]
            return np.argsort(tmp_vs)
        else:  # fixed
            return np.array(range(len(values)))[::-1]

    def _calculate_floor(self, values):
        """Calculate the floor value for stacking"""
        # For negative values, we want them to stack downward from 0
        neg_values = values[values < 0]
        if len(neg_values) == 0:
            return self.kwargs['vsep']/2.0
        
        # Calculate total height needed for negative values
        total_neg_height = abs(sum(neg_values)) + (len(neg_values) - 1) * self.kwargs['vsep']
        return -total_neg_height + self.kwargs['vsep']/2.0

    def _get_transformed_path(self, path_data, pos, floor, ceiling, m_width):
        """Get transformed path with proper scaling and position"""
        # Get original path and its extents
        base_path = path_data['path']
        base_extents = path_data['extents']
        
        # Calculate horizontal stretch factors
        bbox_width = self.kwargs['width'] - 2 * self.kwargs['vpad']
        hstretch_char = bbox_width / base_extents.width
        hstretch_m = bbox_width / m_width
        hstretch = min(hstretch_char, hstretch_m)
        
        # Calculate character width and shift
        char_width = hstretch * base_extents.width
        char_shift = (bbox_width - char_width) / 2.0
        
        # Calculate vertical stretch
        vstretch = (ceiling - floor) / base_extents.height
        
        # Create and apply transformation
        transform = Affine2D()
        transform.translate(tx=-base_extents.xmin, ty=-base_extents.ymin)  # Center first
        transform.scale(hstretch, vstretch)
        transform.translate(
            tx=pos - bbox_width/2.0 + self.kwargs['vpad'] + char_shift,
            ty=floor
        )
        
        final_path = transform.transform_path(base_path)
        return final_path

    def _center_matrix(self, values):
        """Center the values in each position (row) of the matrix"""
        # For each position, subtract the mean of that position
        return values - values.mean(axis=-1, keepdims=True)

    def draw_variability_logo(self, view_window=None, figsize=None, border=True):
        """Draw a variability logo showing all glyphs from all clusters overlaid at each position.
        
        Parameters
        ----------
        view_window : list or tuple, optional
            [start, end] positions to view. If None, show entire logo
        figsize : tuple, optional
            Figure size in inches. If None, use size from initialization
        border : bool, default=True
            Whether to show the axis spines (border)
        """
        # Process all glyphs into logo_data
        logo_data = {'glyphs': []}
        
        # For each position
        for pos in range(self.L):
            # For each cluster
            for cluster_idx in range(self.values.shape[0]):
                values = self.values[cluster_idx, pos]
                ordered_indices = self._get_ordered_indices(values)
                values = values[ordered_indices]
                chars = [str(self.alphabet[i]) for i in ordered_indices]
                
                # Calculate total negative height first
                neg_values = values[values < 0]
                total_neg_height = abs(sum(neg_values)) + (len(neg_values) - 1) * self.kwargs['vsep']
                
                # Handle positive values (stack up from 0)
                floor = self.kwargs['vsep']/2.0
                for value, char in zip(values, chars):
                    if value > 0:
                        ceiling = floor + value
                        
                        path_data = self._path_cache[char]['normal']
                        transformed_path = self._get_transformed_path(
                            path_data, pos, floor, ceiling,
                            self._m_path_cache['extents'].width
                        )
                        
                        logo_data['glyphs'].append({
                            'path': transformed_path,
                            'color': self.rgb_dict[char],
                            'edgecolor': 'none',
                            'edgewidth': 0,
                            'alpha': 1,
                            'floor': floor,
                            'ceiling': ceiling
                        })
                        floor = ceiling + self.kwargs['vsep']
                
                # Handle negative values (stack down from -total_height)
                if len(neg_values) > 0:
                    floor = -total_neg_height - self.kwargs['vsep']/2.0
                    for value, char in zip(values, chars):
                        if value < 0:
                            ceiling = floor + abs(value)
                            
                            path_data = self._path_cache[char]['flipped' if self.kwargs['flip_below'] else 'normal']
                            transformed_path = self._get_transformed_path(
                                path_data, pos, floor, ceiling,
                                self._m_path_cache['extents'].width
                            )
                            
                            logo_data['glyphs'].append({
                                'path': transformed_path,
                                'color': self.rgb_dict[char],
                                'edgecolor': 'none',
                                'edgewidth': 0,
                                'alpha': 1,
                                'floor': floor,
                                'ceiling': ceiling
                            })
                            floor = ceiling + self.kwargs['vsep']
        
        fig, ax = plt.subplots(figsize=figsize if figsize is not None else self.figsize)
        self._draw_single_logo(ax, logo_data, fixed_ylim=True, border=border)
        
        if view_window is not None:
            start, end = view_window
            ax.set_xlim(start-0.5, end-0.5)
        
        plt.tight_layout()
        return fig, ax

    def style_glyphs_in_sequence(self, sequence, color='darkorange'):
        """Style glyphs that match the reference sequence in the specified color, all others dark gray. Disables shade/fade below."""
        if not isinstance(sequence, str):
            raise TypeError('sequence must be a string')
        if len(sequence) != self.L:
            raise ValueError(f'sequence length {len(sequence)} must match logo length {self.L}')
        ref_rgb = to_rgb(color)
        dark_gray = (0.4, 0.4, 0.4)
        for pos in range(self.L):
            ref_char = sequence[pos]
            for logo_idx in self.processed_logos:
                for glyph in self.processed_logos[logo_idx]['glyphs']:
                    if glyph['pos'] == pos:
                        # Disable shade/fade below
                        glyph['alpha'] = 1.0
                        # (if you want to be extra robust, also set any custom keys)
                        if glyph['char'] == ref_char:
                            glyph['color'] = ref_rgb  # Reference glyph
                        else:
                            glyph['color'] = dark_gray  # Non-reference glyph

"""
ARCHITECTURAL DIFFERENCES BETWEEN BATCH_LOGO AND GLYPH_ORIG IMPLEMENTATIONS

This implementation (batch_logo.py) achieves significant performance improvements 
over the original Glyph_orig.py approach through several key optimizations:

1. Path Pre-computation and Caching (see batch_logo.py lines 110-126)
   - Glyph_orig.py: Creates new TextPath objects for each character in each logo
   - batch_logo.py: Pre-computes and caches paths for each character once during initialization
     * Stores both normal and flipped versions with their extents
     * Caches 'M' path for width reference
     * Avoids repeated TextPath creation overhead

2. Transformation Strategy (see Glyph_orig.py lines 451-489 vs batch_logo.py lines 354-401)
   - Glyph_orig.py: Creates individual coordinate systems per character with multiple transforms
   - batch_logo.py: Uses unified coordinate system with optimized transform sequence
     * Centers path using initial extents
     * Applies single combined transformation
     * Maintains visual correctness through careful centering

3. Drawing Strategy (see Glyph_orig.py lines 313-319 vs batch_logo.py lines 271-314)
   - Glyph_orig.py: Adds patches individually to axes
   - batch_logo.py: 
     * Collects all paths for a logo
     * Creates single PatchCollection
     * Uses one draw call per logo
     * Significantly reduces rendering overhead

4. Character Stacking (see Glyph_orig.py lines 462-468 vs batch_logo.py lines 138-195)
   - Glyph_orig.py: Handles flipping through individual Glyph transformations
   - batch_logo.py:
     * Pre-computes negative height requirements
     * Uses pre-flipped paths from cache
     * Maintains correct stacking order efficiently

The result is significantly faster logo generation while maintaining exact visual
parity with the original implementation through careful coordinate system management
and optimized transformation sequences.

"""