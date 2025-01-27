# explicitly set a matplotlib backend if called from python to avoid the
# 'Python is not installed as a framework... error'
import sys
if sys.version_info[0] == 2:
    import matplotlib
    matplotlib.use('TkAgg')

from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D, Bbox
import matplotlib.font_manager as fm
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from logomaker_batch.error_handling import check, handle_errors
from logomaker_batch.colors import get_rgb
import numpy as np
from matplotlib.path import Path
import time
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, Union, List

# Create global list of valid font weights
VALID_FONT_WEIGHT_STRINGS = [
    'ultralight', 'light', 'normal', 'regular', 'book',
    'medium', 'roman', 'semibold', 'demibold', 'demi',
    'bold', 'heavy', 'extra bold', 'black']


def list_font_names():
    """
    Returns a list of valid font_name options for use in Glyph or
    Logo constructors.

    parameters
    ----------
    None.

    returns
    -------
    fontnames: (list)
        List of valid font_name names. This will vary from system to system.

    """
    fontnames_dict = dict([(f.name, f.fname) for f in fm.fontManager.ttflist])
    fontnames = list(fontnames_dict.keys())
    fontnames.append('sans')  # This always exists
    fontnames.sort()
    return fontnames


class TimingContext:
    def __init__(self, name, timing_dict):
        self.name = name
        self.timing_dict = timing_dict
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.timing_dict[self.name] = time.time() - self.start


@dataclass
class GlyphStyle:
    color: Optional[str] = None
    alpha: Optional[float] = None
    shade: float = 0.0
    fade: float = 0.0
    flip: Optional[bool] = None


class Glyph:
    """
    A Glyph represents a character, drawn on a specified axes at a specified
    position, rendered using specified styling such as color and font_name.

    attributes
    ----------

    p: (number)
        x-coordinate value on which to center the Glyph.

    c: (str)
        The character represented by the Glyph.

    floor: (number)
        y-coordinate value where the bottom of the Glyph extends to.
        Must be < ceiling.

    ceiling: (number)
        y-coordinate value where the top of the Glyph extends to.
        Must be > floor.

    ax: (matplotlib Axes object)
        The axes object on which to draw the Glyph.

    width: (number > 0)
        x-coordinate span of the Glyph.

    vpad: (number in [0,1])
        Amount of whitespace to leave within the Glyph bounding box above
        and below the actual Glyph. Specifically, in a glyph with
        height h = ceiling-floor, a margin of size h*vpad/2 will be left blank
        both above and below the rendered character.

    font_name: (str)
        The name of the font to use when rendering the Glyph. This is
        the value passed as the 'family' parameter when calling the
        matplotlib.font_manager.FontProperties constructor.

    font_weight: (str or number)
        The font weight to use when rendering the Glyph. Specifically, this is
        the value passed as the 'weight' parameter in the
        matplotlib.font_manager.FontProperties constructor.
        From matplotlib documentation: "weight: A numeric
        value in the range 0-1000 or one of 'ultralight', 'light',
        'normal', 'regular', 'book', 'medium', 'roman', 'semibold',
        'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'."

    font_size: (number)
        The size of the font to use when rendering the Glyph.

    font_style: (str)
        The style of the font to use when rendering the Glyph.

    color: (matplotlib color)
        Color to use for Glyph face.

    edgecolor: (matplotlib color)
        Color to use for Glyph edge.

    edgewidth: (number >= 0)
        Width of Glyph edge.

    dont_stretch_more_than: (str)
        This parameter limits the amount that a character will be
        horizontally stretched when rendering the Glyph. Specifying a
        wide character such as 'W' corresponds to less potential stretching,
        while specifying a narrow character such as '.' corresponds to more
        stretching.

    flip: (bool)
        If True, the Glyph will be rendered upside down.

    mirror: (bool)
        If True, a mirror image of the Glyph will be rendered.

    zorder: (number)
        Placement of Glyph within the z-stack of ax.

    alpha: (number in [0,1])
        Opacity of the rendered Glyph.

    figsize: ([float, float]):
        The default figure size for the rendered glyph; only used if ax is
        not supplied by the user.

    vsep: (number)
        Vertical separation between glyphs.

    baseline_width: (number)
        Width of the baseline.

    center_values: (bool)
        Whether to center values.

    shade_below: (number in [0,1])
        Amount of shade below the glyph.

    fade_below: (number in [0,1])
        Amount of fade below the glyph.

    fade_probabilities: (bool)
        Whether to fade probabilities.
    """

    # Class-level caches
    _font_cache = {}
    _path_cache = {}
    _m_path_cache = {}
    _bbox_cache = {}  # New cache for bounding boxes
    
    @handle_errors
    def __init__(self, p, c, ax=None, floor=0, ceiling=1, color='black', 
                 flip=False, zorder=0, font_name='Arial', font_weight='normal',
                 font_size=12, font_style='normal', alpha=1.0, vpad=0.0,
                 dont_stretch_more_than='w', mirror=False, rotation=0,
                 edgecolor='none', edgewidth=0.0, alignment='center',
                 background_color='white', background_alpha=0.0,
                 figsize=(10, 2.5), show_spines=True, vsep=0.0,
                 baseline_width=0.5, center_values=False,
                 shade_below=0.0, fade_below=0.0,
                 fade_probabilities=False, **kwargs):
        """Initialize the Glyph with all possible attributes"""
        # Store timing information
        self.timing = {}
        
        # Position and character
        self.p = p
        self.c = c
        self.ax = ax
        
        # Dimensions
        self.floor = floor
        self.ceiling = ceiling
        self.width = 1.0
        self.vpad = vpad
        self.vsep = vsep
        self.figsize = figsize
        self.baseline_width = baseline_width
        
        # Appearance
        self.color = color
        self.edgecolor = edgecolor
        self.edgewidth = edgewidth
        self.alpha = alpha
        self.zorder = zorder
        self.flip = flip
        self.mirror = mirror
        self.rotation = rotation
        self.alignment = alignment
        
        # Background
        self.background_color = background_color
        self.background_alpha = background_alpha
        
        # Font properties
        self.font_name = font_name
        self.font_weight = font_weight
        self.font_size = font_size
        self.font_style = font_style
        
        # Stretching control
        self.dont_stretch_more_than = dont_stretch_more_than
        
        # Display options
        self.show_spines = show_spines
        self.center_values = center_values
        self.shade_below = shade_below
        self.fade_below = fade_below
        self.fade_probabilities = fade_probabilities
        
        # Cache font properties at class level
        if not hasattr(Glyph, '_font_cache'):
            Glyph._font_cache = {}
        
        # Cache the font property for each font_name
        font_name = kwargs.get('font_name', 'sans')
        if font_name not in Glyph._font_cache:
            Glyph._font_cache[font_name] = fm.FontProperties(family=font_name)
        self.prop = Glyph._font_cache[font_name]
        
        # Cache 'M' path for width limiting
        if not hasattr(Glyph, '_M_path_cache'):
            Glyph._M_path_cache = {}
        
        if self.prop.get_name() not in Glyph._M_path_cache:
            msc_path = TextPath((0, 0), 'M', size=1, prop=self.prop)
            Glyph._M_path_cache[self.prop.get_name()] = msc_path
        else:
            msc_path = Glyph._M_path_cache[self.prop.get_name()]
        
        # Additional properties that might be checked
        self.patch = None
        self.transform = None
        self.bbox = None
        self.path = None
        self.transformed_path = None
        self.background_patch = None
        
        # Update any additional attributes passed as kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)
        
        # Perform input validation
        self._input_checks()

        # If ax is not set, set to current axes object
        if self.ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(1, 1))
            self.ax = ax

        # Make patch
        self._make_patch()

    def set_attributes(self, **kwargs):
        """
        Safe way to set the attributes of a Glyph object

        parameters
        ----------
        **kwargs:
            Attributes and their values.
        """

        # remove drawn patch
        if (self.patch is not None) and (self.patch.axes is not None):
            self.patch.remove()

        # set each attribute passed by user
        for key, value in kwargs.items():

            # if key corresponds to a color, convert to rgb
            if key in ('color', 'edgecolor'):
                value = to_rgb(value)

            # save variable name
            self.__dict__[key] = value

        # remake patch
        self._make_patch()

    def draw(self):
        """Draw the glyph using its current attributes"""
        path = self._make_patch()
        if path is not None:
            patch = PathPatch(path,
                            facecolor=self.color,
                            edgecolor=self.edgecolor,
                            linewidth=self.edgewidth,
                            alpha=self.alpha,
                            zorder=self.zorder)
            self.ax.add_patch(patch)
            self.patch = patch

    def _get_raw_vertices(self):
        """Get untransformed vertices"""
        path = TextPath((0, 0), self.c, size=1, prop=self.prop)
        return path.vertices

    def _draw_transformed(self, vertices):
        """Draw glyph using pre-transformed vertices"""
        path = TextPath((0, 0), self.c, size=1, prop=self.prop)
        path._vertices = vertices
        self.patch = PathPatch(path,
                            facecolor=self.color,
                            edgecolor=self.edgecolor,
                            linewidth=self.edgewidth,
                            zorder=self.zorder,
                            alpha=self.alpha)
        self.ax.add_patch(self.patch)

    def _make_patch(self):
        """Returns an appropriately scaled patch object corresponding to the Glyph."""
        height = self.ceiling - self.floor
        if height == 0.0:
            self.patch = None
            return None

        # Get or create text path
        cache_key = (self.c, self.font_name)
        if cache_key not in self._path_cache:
            tmp_path = TextPath((0, 0), self.c, size=1, prop=self.prop)
            self._path_cache[cache_key] = tmp_path  # Store original path
            self._bbox_cache[cache_key] = tmp_bbox = tmp_path.get_extents()
        else:
            tmp_path = self._path_cache[cache_key]  # Use cached path directly
            tmp_bbox = self._bbox_cache[cache_key]

        # Get or create M path and its bbox
        m_key = ('M', self.font_name)
        if m_key not in self._path_cache:
            msc_path = TextPath((0, 0), 'M', size=1, prop=self.prop)
            self._path_cache[m_key] = msc_path
            self._bbox_cache[m_key] = msc_bbox = msc_path.get_extents()
        else:
            msc_path = self._path_cache[m_key]
            msc_bbox = self._bbox_cache[m_key]

        # Create target bounding box
        bbox = Bbox([[self.p - self.width/2.0 + self.vpad, self.floor],
                    [self.p + self.width/2.0 - self.vpad, self.ceiling]])
        
        # Calculate stretching factors more efficiently
        widths = np.array([tmp_bbox.width, msc_bbox.width])
        hstretches = bbox.width / widths
        hstretch = np.min(hstretches)
        char_width = hstretch * tmp_bbox.width
        char_shift = (bbox.width - char_width) / 2.0
        vstretch = bbox.height / tmp_bbox.height
        
        # Apply final transformation
        transformation = Affine2D() \
            .translate(tx=-tmp_bbox.xmin, ty=-tmp_bbox.ymin) \
            .scale(sx=hstretch, sy=vstretch) \
            .translate(tx=bbox.xmin + char_shift, ty=bbox.ymin)
        final_path = transformation.transform_path(tmp_path)
        
        return final_path

    def _input_checks(self):

        """
        check input parameters in the Logo constructor for correctness
        """

        from numbers import Number
        # validate p
        check(isinstance(int(self.p), (float, int)),
              'type(p) = %s must be a number' % type(self.p))

        # check c is of type str
        check(isinstance(self.c, str),
              'type(c) = %s; must be of type str ' %
              type(self.c))

        # validate floor
        check(isinstance(self.floor, (float, int)),
              'type(floor) = %s must be a number' % type(self.floor))
        self.floor = float(self.floor)

        # validate ceiling
        check(isinstance(self.ceiling, (float, int)),
              'type(ceiling) = %s must be a number' % type(self.ceiling))
        self.ceiling = float(self.ceiling)

        # check floor <= ceiling
        check(self.floor <= self.ceiling,
              'must have floor <= ceiling. Currently, '
              'floor=%f, ceiling=%f' % (self.floor, self.ceiling))

        # check ax
        check((self.ax is None) or isinstance(self.ax, Axes),
              'ax must be either a matplotlib Axes object or None.')

        # validate width
        check(isinstance(self.width, (float, int)),
              'type(width) = %s; must be of type float or int ' %
              type(self.width))
        check(self.width > 0, "width = %d must be > 0 " %
              self.width)

        # validate vpad
        check(isinstance(self.vpad, (float, int)),
              'type(vpad) = %s; must be of type float or int ' %
              type(self.vpad))
        check(0 <=self.vpad <1, "vpad = %d must be >= 0 and < 1 " %
              self.vpad)

        # validate font_name
        check(isinstance(self.font_name, str),
              'type(font_name) = %s must be of type str' % type(self.font_name))

        # check font_weight
        check(isinstance(self.font_weight, (str, int)),
              'type(font_weight) = %s should either be a string or an int' %
              (type(self.font_weight)))
        if isinstance(self.font_weight, str):
            check(self.font_weight in VALID_FONT_WEIGHT_STRINGS,
                  'font_weight must be one of %s' % VALID_FONT_WEIGHT_STRINGS)
        elif isinstance(self.font_weight, int):
            check(0 <= self.font_weight <= 1000,
                  'font_weight must be in range [0,1000]')

        # check color safely
        self.color = get_rgb(self.color)

        # validate edgecolor safely
        self.edgecolor = get_rgb(self.edgecolor)

        # Check that edgewidth is a number
        check(isinstance(self.edgewidth, (float, int)),
              'type(edgewidth) = %s must be a number' % type(self.edgewidth))
        self.edgewidth = float(self.edgewidth)

        # Check that edgewidth is nonnegative
        check(self.edgewidth >= 0,
              ' edgewidth must be >= 0; is %f' % self.edgewidth)

        # check dont_stretch_more_than is of type str
        check(isinstance(self.dont_stretch_more_than, str),
              'type(dont_stretch_more_than) = %s; must be of type str ' %
              type(self.dont_stretch_more_than))

        # check that dont_stretch_more_than is a single character
        check(len(self.dont_stretch_more_than)==1,
              'dont_stretch_more_than must have length 1; '
              'currently len(dont_stretch_more_than)=%d' %
              len(self.dont_stretch_more_than))

        # check that flip is a boolean
        check(isinstance(self.flip, (bool, np.bool_)),
              'type(flip) = %s; must be of type bool ' % type(self.flip))
        self.flip = bool(self.flip)

        # check that mirror is a boolean
        check(isinstance(self.mirror, (bool, np.bool_)),
              'type(mirror) = %s; must be of type bool ' % type(self.mirror))
        self.mirror = bool(self.mirror)

        # validate zorder
        if self.zorder is not None :
            check(isinstance(self.zorder, (float, int)),
                  'type(zorder) = %s; must be of type float or int ' %
                  type(self.zorder))

        # Check alpha is a number
        check(isinstance(self.alpha, (float, int)),
              'type(alpha) = %s must be a float or int' %
              type(self.alpha))
        self.alpha = float(self.alpha)

        # Check 0 <= alpha <= 1.0
        check(0 <= self.alpha <= 1.0,
              'alpha must be between 0.0 and 1.0 (inclusive)')

        # validate that figsize is array=like
        check(isinstance(self.figsize, (tuple, list, np.ndarray)),
              'type(figsize) = %s; figsize must be array-like.' %
              type(self.figsize))
        self.figsize = tuple(self.figsize) # Just to pin down variable type.

        # validate length of figsize
        check(len(self.figsize) == 2, 'figsize must have length two.')

        # validate that each element of figsize is a number
        check(all([isinstance(n, (int, float)) and n > 0
                   for n in self.figsize]),
              'all elements of figsize array must be numbers > 0.')

    def _create_path_from_vertices(self, vertices):
        """Create a path from pre-transformed vertices"""
        return Path(vertices)

    def _get_transformed_path(self):
        """Get the transformed path with timing information"""
        timing = {}
        
        # Get or create text path
        t0 = time.time()
        cache_key = (self.c, self.font_name)
        cache_hit = cache_key in self._path_cache
        if not cache_hit:
            print(f"Cache miss for {cache_key}")
            tmp_path = TextPath((0, 0), self.c, size=1, prop=self.prop)
            self._path_cache[cache_key] = tmp_path
        tmp_path = self._path_cache[cache_key]
        timing['text_path'] = time.time() - t0
        timing['cache_hit'] = cache_hit
        
        # Get or create M path
        t0 = time.time()
        if self.font_name not in self._m_path_cache:
            msc_path = TextPath((0, 0), 'M', size=1, prop=self.prop)
            self._m_path_cache[self.font_name] = msc_path
        msc_path = self._m_path_cache[self.font_name]
        timing['m_path'] = time.time() - t0
        
        # Create bounding box
        t0 = time.time()
        bbox = Bbox([[self.p - self.width/2.0 + self.vpad, self.floor],
                    [self.p + self.width/2.0 - self.vpad, self.ceiling]])
        timing['bbox_create'] = time.time() - t0
        
        # Apply flip/mirror transformations
        t0 = time.time()
        if self.flip:
            transformation = Affine2D().scale(sx=1, sy=-1)
            tmp_path = transformation.transform_path(tmp_path)
        if self.mirror:
            transformation = Affine2D().scale(sx=-1, sy=1)
            tmp_path = transformation.transform_path(tmp_path)
        timing['flip_mirror'] = time.time() - t0
        
        # Calculate stretching factors more efficiently
        t0 = time.time()
        widths = np.array([tmp_path.get_extents().width, msc_path.get_extents().width])
        hstretches = bbox.width / widths
        hstretch = np.min(hstretches)
        char_width = hstretch * tmp_path.get_extents().width
        char_shift = (bbox.width - char_width) / 2.0
        vstretch = bbox.height / tmp_path.get_extents().height
        timing['stretch_calc'] = time.time() - t0
        
        # Apply final transformation
        t0 = time.time()
        transformation = Affine2D() \
            .translate(tx=-tmp_path.get_extents().xmin, ty=-tmp_path.get_extents().ymin) \
            .scale(sx=hstretch, sy=vstretch) \
            .translate(tx=bbox.xmin + char_shift, ty=bbox.ymin)
        final_path = transformation.transform_path(tmp_path)
        timing['transform_apply'] = time.time() - t0
        
        self.timing.update(timing)
        return final_path

    @classmethod
    @lru_cache(maxsize=128)
    def _create_text_path(cls, char, font_name, prop):
        path = TextPath((0, 0), char, size=1, prop=prop)
        bbox = path.get_extents()
        return path, bbox


