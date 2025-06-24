'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright Evan Seitz 2023-2025
Cold Spring Harbor Laboratory
Contact: evan.e.seitz@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Minimum installation:
    conda create --name seam-gui python==3.8*
    pip install --upgrade pip
    pip install PyQt5
    pip3 install --user psutil
    pip install biopython
    pip install scipy
    pip install seaborn
    pip install -U scikit-learn
    pip install pysam
    pip install seam-nn
    pip install matplotlib==3.6

Installation notes:
    - Make sure matplotlib==3.6 is the last package installed
    - The seam-gui environment should be separate from the seam-nn environment due to matplotlib version conflicts
    - The seam-gui environment should be activated before installing packages and running the GUI
'''


import sys, os
sys.dont_write_bytecode = True
import copy
import psutil, logging # 'pip3 install --user psutil'
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton,\
     QGridLayout, QLabel, QSplitter, QFrame, QLineEdit, QFileDialog, QStyleFactory,\
     QMessageBox, QCheckBox, QSpinBox, QDoubleSpinBox, QAbstractSpinBox, QComboBox
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.path as pltPath
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator # integer ticks
import pandas as pd
from Bio import motifs # 'pip install biopython'
from scipy import spatial # 'pip install scipy'
from scipy.cluster import hierarchy
from tqdm import tqdm
if 0: # enable local SEAM imports TODO
    sys.path.insert(0, os.path.abspath('.'))
    from logomaker_batch.batch_logo import BatchLogo
    from clusterer import Clusterer
    from meta_explainer import MetaExplainer
    import utils
else: # enable SEAM imports from pip install
    from seam.logomaker_batch.batch_logo import BatchLogo
    from seam import Clusterer
    from seam import MetaExplainer
    from seam import utils
def warn(*args, **kwargs): # ignore matplotlib deprecation warnings
    pass
import warnings
warnings.warn = warn


################################################################################
# Global assets

py_dir = os.path.dirname(os.path.realpath(__file__)) # python file location
parent_dir = os.path.dirname(py_dir)
icon_dir = os.path.join(parent_dir, 'docs/_static/gui_icons')

progname = 'SEAM Interactive Interpretability Tool'
font_standard = QtGui.QFont('Arial', 12)

dpi = 200 # or: plt.rcParams['savefig.dpi'] = dpi

################################################################################
# Global widgets

class QVLine(QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)

################################################################################
# File imports tab

class Imports(QtWidgets.QWidget):
    mave_fname = ''
    maps_fname = ''
    embedding_fname = ''
    linkage_fname = ''
    clusters = ''
    mave_col_names = ''
    ref_full = ''
    ref_idx = 0
    hamming_orig = ''
    current_tab = 0
    cluster_col = 'Cluster'
    cluster_sort_method = 'Median activity'  # Store selected cluster sorting method
    psi_1 = 0
    psi_2 = 1
    linkage = None  # Store linkage matrix if loaded
    batch_logo_instances = {}  # Store BatchLogo instances for each cluster and scale type
    batch_logo_type = 'average'  # Type of logo to generate: 'average', 'pwm', 'enrichment'
    batch_logo_yscale = 'adaptive'  # Y-axis scaling: 'adaptive' or 'fixed'
    imports_changed = False  # Track if imports have been changed since last confirmation
    imports_confirmed = False  # Track if imports have been initially confirmed
    imports_warning_acknowledged = False  # Track if the warning popup has been shown and acknowledged
    background_separation = False  # Track background separation setting
    mutation_rate = 0.10  # Track mutation rate setting
    adaptive_background_scaling = True  # Track adaptive background scaling setting
    entropy_multiplier = 0.5  # Track entropy multiplier setting

    def __init__(self, parent=None):
        super(Imports, self).__init__(parent)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(20,20,20,20) # W, N, E, S
        self.grid.setSpacing(10)

        # Track required file states
        self.has_embedding = False
        self.has_clusters = False

        # --- Widget creation section ---
        self.label_mave = QLabel('MAVE dataset')
        self.label_mave.setFont(font_standard)
        self.label_mave.setMargin(20)
        self.entry_mave = QLineEdit('Filename')
        self.entry_mave.setDisabled(True)
        self.browse_mave = QPushButton('          Browse          ', self)
        self.browse_mave.clicked.connect(self.load_mave)
        self.browse_mave.setToolTip('Accepted format: <i>.csv</i>')

        self.label_ref = QLabel('Reference sequence:')
        self.label_ref.setFont(font_standard)
        self.label_ref.setMargin(20)
        Imports.entry_ref = QLineEdit('')
        Imports.entry_ref.setReadOnly(True)
        Imports.entry_ref.setDisabled(True)
        Imports.combo_ref = QComboBox(self)
        Imports.combo_ref.setDisabled(True)
        Imports.combo_ref.addItem('First row')
        Imports.combo_ref.addItem('Custom')
        Imports.combo_ref.addItem('None')
        Imports.combo_ref.currentTextChanged.connect(self.select_ref)
        Imports.combo_ref.setToolTip('Define a reference sequence for the MAVE dataset.')
        Imports.combo_ref.setFocus(True)

        self.label_maps = QLabel('Attribution maps')
        self.label_maps.setFont(font_standard)
        self.label_maps.setMargin(20)
        self.entry_maps = QLineEdit('Filename')
        self.entry_maps.setDisabled(True)
        self.browse_maps = QPushButton('          Browse          ', self)
        self.browse_maps.clicked.connect(self.load_maps)
        self.browse_maps.setToolTip('Accepted formats: <i>.npy</i> or <i>.npz</i> (compressed)')

        class DoubleSpinBox(QtWidgets.QDoubleSpinBox):
            def changeEvent(self, e):
                if e.type() == QtCore.QEvent.EnabledChange:
                    self.lineEdit().setVisible(self.isEnabled())
                return super().changeEvent(e)

        self.label_start = QLabel('Map start position:')
        self.label_start.setFont(font_standard)
        self.label_start.setMargin(20)
        Imports.startbox_cropped = DoubleSpinBox(self)
        Imports.startbox_cropped.setButtonSymbols(QAbstractSpinBox.NoButtons)
        Imports.startbox_cropped.setDecimals(0)
        Imports.startbox_cropped.setMinimum(0)
        Imports.startbox_cropped.setFont(font_standard)
        Imports.startbox_cropped.setDisabled(True)
        Imports.startbox_cropped.valueChanged.connect(self.cropped_signal)
        Imports.startbox_cropped.setToolTip('Provide the new start position with respect to MAVE sequences.')

        self.label_stop = QLabel('Map stop position:')
        self.label_stop.setFont(font_standard)
        self.label_stop.setMargin(20)
        Imports.stopbox_cropped = DoubleSpinBox(self)
        Imports.stopbox_cropped.setButtonSymbols(QAbstractSpinBox.NoButtons)
        Imports.stopbox_cropped.setDecimals(0)
        Imports.stopbox_cropped.setMinimum(1)
        Imports.stopbox_cropped.setFont(font_standard)
        Imports.stopbox_cropped.setDisabled(True)
        Imports.stopbox_cropped.valueChanged.connect(self.cropped_signal)
        Imports.stopbox_cropped.setToolTip('Provide the new stop position with respect to MAVE sequences.')
   
        self.label_embedding = QLabel('Embedding')
        self.label_embedding.setFont(font_standard)
        self.label_embedding.setMargin(20)
        Imports.checkbox_embedding = QCheckBox(self)
        Imports.checkbox_embedding.setToolTip('Check to enable embedding file import for clustering.')
        Imports.checkbox_embedding.stateChanged.connect(self.toggle_embedding)
        self.entry_embedding = QLineEdit('Filename')
        self.entry_embedding.setDisabled(True)
        self.browse_embedding = QPushButton('          Browse          ', self)
        self.browse_embedding.clicked.connect(self.load_embedding)
        self.browse_embedding.setToolTip('Accepted formats: <i>.npy</i>')

        self.label_linkage = QLabel('Linkage')
        self.label_linkage.setFont(font_standard)
        self.label_linkage.setMargin(20)
        Imports.checkbox_linkage = QCheckBox(self)
        Imports.checkbox_linkage.setToolTip('Check to enable linkage file import for clustering.')
        Imports.checkbox_linkage.stateChanged.connect(self.toggle_linkage)
        self.entry_linkage = QLineEdit('Filename')
        self.entry_linkage.setDisabled(True)
        self.browse_linkage = QPushButton('          Browse          ', self)
        self.browse_linkage.clicked.connect(self.load_clusters_or_linkage)
        self.browse_linkage.setToolTip('Accepted formats: <i>.npy</i>')

        # Add linkage cutting controls
        self.label_cut_criterion = QLabel('Cut criterion:')
        self.label_cut_criterion.setFont(font_standard)
        self.label_cut_criterion.setMargin(20)

        Imports.combo_cut_criterion = QComboBox(self)
        Imports.combo_cut_criterion.addItem('maxclust')
        Imports.combo_cut_criterion.addItem('distance')

        self.label_cut_param = QLabel('Number of clusters:')
        self.label_cut_param.setFont(font_standard)
        self.label_cut_param.setMargin(20)

        Imports.spin_cut_param = QtWidgets.QSpinBox(self)
        Imports.spin_cut_param.setMinimum(2)
        Imports.spin_cut_param.setValue(30)
        Imports.spin_cut_param.valueChanged.connect(lambda: self.mark_imports_changed())

        # Cluster sorting dropdown
        self.label_linkage_sort_method = QLabel('Cluster sorting:')
        self.label_linkage_sort_method.setFont(font_standard)
        self.label_linkage_sort_method.setMargin(20)
        self.combo_linkage_sort_method = QComboBox(self)
        self.combo_linkage_sort_method.addItem('Median activity')
        self.combo_linkage_sort_method.addItem('No reordering')
        self.combo_linkage_sort_method.setCurrentIndex(0)
        self.combo_linkage_sort_method.setToolTip('Choose how to sort clusters in summary matrix visualizations.')
        def set_sort_method(value):
            Imports.cluster_sort_method = value
            self.mark_imports_changed()
        self.combo_linkage_sort_method.currentTextChanged.connect(set_sort_method)

        # Add embedding clustering controls (mirroring linkage controls)
        self.label_embedding_method = QLabel('Clustering method:')
        self.label_embedding_method.setFont(font_standard)
        self.label_embedding_method.setMargin(20)

        Imports.combo_embedding_method = QComboBox(self)
        Imports.combo_embedding_method.addItem('kmeans')
        Imports.combo_embedding_method.addItem('dbscan')

        def toggle_cluster_spinbox():
            method = Imports.combo_embedding_method.currentText()
            if method == 'kmeans':
                self.label_embedding_clusters.setVisible(True)
                Imports.spin_embedding_clusters.setVisible(True)
            elif method == 'dbscan':
                self.label_embedding_clusters.setVisible(False)
                Imports.spin_embedding_clusters.setVisible(False)
            self.mark_imports_changed()

        Imports.combo_embedding_method.currentTextChanged.connect(toggle_cluster_spinbox)

        self.label_embedding_clusters = QLabel('Number of clusters:')
        self.label_embedding_clusters.setFont(font_standard)
        self.label_embedding_clusters.setMargin(20)

        Imports.spin_embedding_clusters = QtWidgets.QSpinBox(self)
        Imports.spin_embedding_clusters.setMinimum(2)
        Imports.spin_embedding_clusters.setMaximum(1000)
        Imports.spin_embedding_clusters.setValue(200)
        Imports.spin_embedding_clusters.valueChanged.connect(lambda: self.mark_imports_changed())

        # Add embedding cluster sorting dropdown (separate widget but same functionality)
        self.label_embedding_sort_method = QLabel('Cluster sorting:')
        self.label_embedding_sort_method.setFont(font_standard)
        self.label_embedding_sort_method.setMargin(20)
        self.combo_embedding_sort_method = QComboBox(self)
        self.combo_embedding_sort_method.addItem('Median activity')
        self.combo_embedding_sort_method.addItem('No reordering')
        self.combo_embedding_sort_method.setCurrentIndex(0)
        self.combo_embedding_sort_method.setToolTip('Choose how to sort clusters in summary matrix visualizations.')
        self.combo_embedding_sort_method.currentTextChanged.connect(set_sort_method)

        # Add Additional Options section
        self.label_additional = QLabel('Additional options')
        self.label_additional.setFont(font_standard)
        self.label_additional.setMargin(20)

        # Background separation checkbox
        self.label_background_separation = QLabel('Background separation')
        self.label_background_separation.setFont(font_standard)
        self.label_background_separation.setMargin(20)
        
        Imports.checkbox_background_separation = QCheckBox(self)
        Imports.checkbox_background_separation.setToolTip('Enable background separation for analysis of local sequence library.')
        def update_background_separation():
            Imports.background_separation = Imports.checkbox_background_separation.isChecked()
            # Show/hide background-related widgets based on background separation state
            self.label_mutation_rate.setVisible(Imports.background_separation)
            Imports.spin_mutation_rate.setVisible(Imports.background_separation)
            self.label_adaptive_scaling.setVisible(Imports.background_separation)
            Imports.checkbox_adaptive_scaling.setVisible(Imports.background_separation)
            self.label_entropy_multiplier.setVisible(Imports.background_separation)
            Imports.spin_entropy_multiplier.setVisible(Imports.background_separation)
            self.mark_imports_changed()
        Imports.checkbox_background_separation.stateChanged.connect(update_background_separation)

        # Mutation rate label and spinbox
        self.label_mutation_rate = QLabel('Mutation rate:')
        self.label_mutation_rate.setFont(font_standard)
        self.label_mutation_rate.setMargin(20)

        Imports.spin_mutation_rate = QtWidgets.QDoubleSpinBox(self)
        Imports.spin_mutation_rate.setMinimum(0.0)
        Imports.spin_mutation_rate.setMaximum(1.0)
        Imports.spin_mutation_rate.setValue(0.10)
        Imports.spin_mutation_rate.setDecimals(2)
        Imports.spin_mutation_rate.setSingleStep(0.01)
        def update_mutation_rate():
            Imports.mutation_rate = Imports.spin_mutation_rate.value()
            self.mark_imports_changed()
        Imports.spin_mutation_rate.valueChanged.connect(update_mutation_rate)
        Imports.spin_mutation_rate.setToolTip('Set the mutation rate for analysis (0.0 to 1.0).')

        # Adaptive scaling label and checkbox
        self.label_adaptive_scaling = QLabel('Adaptive scaling:')
        self.label_adaptive_scaling.setFont(font_standard)
        self.label_adaptive_scaling.setMargin(20)

        Imports.checkbox_adaptive_scaling = QCheckBox(self)
        Imports.checkbox_adaptive_scaling.setToolTip('Enable cluster-specific background scaling for better background separation.')
        Imports.checkbox_adaptive_scaling.setChecked(True)
        def update_adaptive_scaling():
            Imports.adaptive_background_scaling = Imports.checkbox_adaptive_scaling.isChecked()
            self.mark_imports_changed()
        Imports.checkbox_adaptive_scaling.stateChanged.connect(update_adaptive_scaling)

        # Entropy multiplier label and spinbox
        self.label_entropy_multiplier = QLabel('Entropy multiplier:')
        self.label_entropy_multiplier.setFont(font_standard)
        self.label_entropy_multiplier.setMargin(20)

        Imports.spin_entropy_multiplier = QtWidgets.QDoubleSpinBox(self)
        Imports.spin_entropy_multiplier.setMinimum(0.1)
        Imports.spin_entropy_multiplier.setMaximum(1.0)
        Imports.spin_entropy_multiplier.setValue(0.5)
        Imports.spin_entropy_multiplier.setDecimals(1)
        Imports.spin_entropy_multiplier.setSingleStep(0.1)
        def update_entropy_multiplier():
            Imports.entropy_multiplier = Imports.spin_entropy_multiplier.value()
            self.mark_imports_changed()
        Imports.spin_entropy_multiplier.valueChanged.connect(update_entropy_multiplier)
        Imports.spin_entropy_multiplier.setToolTip('Control background position identification stringency (0.1 to 1.0). Lower values are more stringent.')

        # Update cut parameter label and spinbox when criterion changes
        def update_cut_param():
            crit = Imports.combo_cut_criterion.currentText()
            if crit == 'maxclust':
                self.label_cut_param.setText('Number of clusters:')
                # Update the existing spinbox to be an integer spinbox
                if not isinstance(self.spin_cut_param, QtWidgets.QSpinBox):
                    # Remove the old widget from the grid first
                    self.grid.removeWidget(self.spin_cut_param)
                    val = self.spin_cut_param.value()
                    self.spin_cut_param = QtWidgets.QSpinBox(self)
                    self.spin_cut_param.setMinimum(2)
                    self.spin_cut_param.setValue(int(val))
                    self.spin_cut_param.setSingleStep(1)
                    self.spin_cut_param.valueChanged.connect(lambda: self.mark_imports_changed())
                    self.grid.addWidget(self.spin_cut_param, 7, 5, 1, 1, QtCore.Qt.AlignLeft)
            else:
                self.label_cut_param.setText('Distance threshold:')
                # Update the existing spinbox to be a double spinbox
                if not isinstance(self.spin_cut_param, QtWidgets.QDoubleSpinBox):
                    # Remove the old widget from the grid first
                    self.grid.removeWidget(self.spin_cut_param)
                    val = self.spin_cut_param.value()
                    self.spin_cut_param = QtWidgets.QDoubleSpinBox(self)
                    self.spin_cut_param.setMinimum(0.0)
                    self.spin_cut_param.setDecimals(3)
                    self.spin_cut_param.setSingleStep(0.01)
                    self.spin_cut_param.setValue(float(val))
                    self.spin_cut_param.valueChanged.connect(lambda: self.mark_imports_changed())
                    self.grid.addWidget(self.spin_cut_param, 7, 5, 1, 1, QtCore.Qt.AlignLeft)            
            self.mark_imports_changed()
        self.combo_cut_criterion.currentTextChanged.connect(update_cut_param)

        # End of tab completion
        self.label_empty = QLabel('')
        self.line_L = QLabel('') # aesthetic line left
        self.line_L.setFont(font_standard)
        self.line_L.setMargin(0)
        self.line_L.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        self.line_R = QLabel('') # aesthetic line right
        self.line_R.setFont(font_standard)
        self.line_R.setMargin(0)
        self.line_R.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        Imports.btn_process_imports = QPushButton('Confirm Imports', self)
        Imports.btn_process_imports.setToolTip('Either Embedding OR Linkage must be selected (but not both).')
        
        # Track if imports have been initially confirmed
        self.imports_confirmed = False

        # First row
        self.grid.addWidget(self.label_mave, 0, 1, 1, 1, QtCore.Qt.AlignLeft)
        self.grid.addWidget(self.entry_mave, 0, 2, 1, 6)
        self.grid.addWidget(self.browse_mave, 0, 8, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # First row, subrow
        self.grid.addWidget(self.label_ref, 1, 2, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(Imports.entry_ref, 1, 3, 1, 4)
        self.grid.addWidget(Imports.combo_ref, 1, 7, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # Second row
        self.grid.addWidget(self.label_maps, 2, 1, 1, 1, QtCore.Qt.AlignLeft)
        self.grid.addWidget(self.entry_maps, 2, 2, 1, 6)
        self.grid.addWidget(self.browse_maps, 2, 8, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # Second row, subrow
        self.grid.addWidget(self.label_start, 3, 2, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(Imports.startbox_cropped, 3, 3, 1, 1, QtCore.Qt.AlignLeft)
        self.grid.addWidget(self.label_stop, 3, 4, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(Imports.stopbox_cropped, 3, 5, 1, 1, QtCore.Qt.AlignLeft)
        # Third row: Embedding
        self.grid.addWidget(Imports.checkbox_embedding, 4, 0, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(self.label_embedding, 4, 1, 1, 1, QtCore.Qt.AlignLeft)
        self.grid.addWidget(self.entry_embedding, 4, 2, 1, 6)
        self.grid.addWidget(self.browse_embedding, 4, 8, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # Fourth row: Embedding sub-widgets (indented under embedding row)
        self.grid.addWidget(self.label_embedding_method, 5, 2, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(Imports.combo_embedding_method, 5, 3, 1, 1, QtCore.Qt.AlignLeft)
        self.grid.addWidget(self.label_embedding_clusters, 5, 4, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(Imports.spin_embedding_clusters, 5, 5, 1, 1, QtCore.Qt.AlignLeft)
        self.grid.addWidget(self.label_embedding_sort_method, 5, 6, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(self.combo_embedding_sort_method, 5, 7, 1, 1, QtCore.Qt.AlignLeft)
        # Fifth row: Linkage
        self.grid.addWidget(Imports.checkbox_linkage, 6, 0, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(self.label_linkage, 6, 1, 1, 1, QtCore.Qt.AlignLeft)
        self.grid.addWidget(self.entry_linkage, 6, 2, 1, 6)
        self.grid.addWidget(self.browse_linkage, 6, 8, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # Sixth row: Linkage sub-widgets (indented under linkage row)
        self.grid.addWidget(self.label_cut_criterion, 7, 2, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(Imports.combo_cut_criterion, 7, 3, 1, 1, QtCore.Qt.AlignLeft)
        self.grid.addWidget(self.label_cut_param, 7, 4, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(Imports.spin_cut_param, 7, 5, 1, 1, QtCore.Qt.AlignLeft)
        self.grid.addWidget(self.label_linkage_sort_method, 7, 6, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(self.combo_linkage_sort_method, 7, 7, 1, 1, QtCore.Qt.AlignLeft)
        # Small aesthetic line between sections
        self.line_mid = QLabel('') # aesthetic line middle
        self.line_mid.setFont(font_standard)
        self.line_mid.setMargin(0)
        self.line_mid.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        self.grid.addWidget(self.line_mid, 8, 1, 1, 8, QtCore.Qt.AlignVCenter)
        # Seventh row: Additional options
        self.grid.addWidget(self.label_additional, 9, 1, 1, 1, QtCore.Qt.AlignLeft)
        # Eighth row: Additional options sub-widgets (indented under additional options row)
        self.grid.addWidget(Imports.checkbox_background_separation, 10, 1, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(self.label_background_separation, 10, 2, 1, 1, QtCore.Qt.AlignLeft)
        self.grid.addWidget(self.label_mutation_rate, 10, 3, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(Imports.spin_mutation_rate, 10, 4, 1, 1, QtCore.Qt.AlignLeft)
        self.grid.addWidget(self.label_adaptive_scaling, 10, 5, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(Imports.checkbox_adaptive_scaling, 10, 6, 1, 1, QtCore.Qt.AlignLeft)
        self.grid.addWidget(self.label_entropy_multiplier, 10, 7, 1, 1, QtCore.Qt.AlignRight)
        self.grid.addWidget(Imports.spin_entropy_multiplier, 10, 8, 1, 1, QtCore.Qt.AlignLeft)
        # Final section for submitting imports
        self.grid.addWidget(self.label_empty, 11, 2, 5, 8, QtCore.Qt.AlignRight)
        self.grid.addWidget(self.line_L, 16, 0, 1, 4, QtCore.Qt.AlignVCenter)
        self.grid.addWidget(self.line_R, 16, 6, 1, 4, QtCore.Qt.AlignVCenter)
        self.grid.addWidget(Imports.btn_process_imports, 16, 4, 1, 2)

        self.setLayout(self.grid)

        # Initialize all subwidgets in disabled state (except browse buttons)
        self.update_mave_subwidgets_state(False)
        self.update_maps_subwidgets_state(False)
        self.update_cluster_subwidgets_state(False)
        
        # Initialize embedding sub-widgets as disabled (following same pattern as MAVE/maps)
        self.entry_embedding.setDisabled(True)
        self.label_embedding_method.setEnabled(False)
        self.combo_embedding_method.setEnabled(False)
        self.label_embedding_clusters.setEnabled(False)
        self.spin_embedding_clusters.setEnabled(False)
        self.label_embedding_sort_method.setEnabled(False)
        self.combo_embedding_sort_method.setEnabled(False)

        # Set embedding as default checked option
        Imports.checkbox_embedding.setChecked(True)
        Imports.checkbox_linkage.setChecked(False)
        
        # Enable embedding browse button by default (since embedding is checked)
        self.browse_embedding.setEnabled(True)
        
        # Ensure linkage widgets are disabled
        self.entry_linkage.setEnabled(False)
        self.browse_linkage.setEnabled(False)
        self.update_cluster_subwidgets_state(False)

        # Set initial state for embedding clustering method
        toggle_cluster_spinbox()

        # Initialize mutation rate widgets as hidden (background separation not checked by default)
        self.label_mutation_rate.setVisible(False)
        Imports.spin_mutation_rate.setVisible(False)
        self.label_adaptive_scaling.setVisible(False)
        Imports.checkbox_adaptive_scaling.setVisible(False)
        self.label_entropy_multiplier.setVisible(False)
        Imports.spin_entropy_multiplier.setVisible(False)


    def update_mave_subwidgets_state(self, enabled):
        """Enable or disable MAVE subwidgets based on whether MAVE file is loaded"""
        Imports.entry_ref.setReadOnly(not enabled)
        Imports.combo_ref.setDisabled(not enabled)
        Imports.startbox_cropped.setEnabled(enabled)
        Imports.stopbox_cropped.setEnabled(enabled)
        if not enabled:
            Imports.entry_ref.setText('')
            Imports.combo_ref.setCurrentIndex(0)
            Imports.startbox_cropped.setValue(0)
            Imports.stopbox_cropped.setValue(1)
            Imports.stopbox_cropped.setSuffix('')

    def update_maps_subwidgets_state(self, enabled):
        """Enable or disable maps subwidgets based on whether maps file is loaded"""
        # Don't disable start/stop position widgets - they should be controlled by MAVE file
        # Only enable/disable embedding widgets since they depend on maps
        if not enabled:
            self.entry_embedding.setText('Filename')
            self.entry_embedding.setDisabled(True)
            self.has_embedding = False
            self.mark_imports_changed()

    def update_cluster_subwidgets_state(self, enabled):
        """Enable or disable cluster subwidgets based on whether cluster file is loaded"""
        self.combo_cut_criterion.setEnabled(enabled)
        self.spin_cut_param.setEnabled(enabled)
        self.combo_linkage_sort_method.setEnabled(enabled)

    def toggle_embedding(self):
        """Handle embedding checkbox state change"""
        if Imports.checkbox_embedding.isChecked():
            # Uncheck linkage checkbox if embedding is checked
            Imports.checkbox_linkage.setChecked(False)
            # Clear linkage filename when switching to embedding
            Imports.linkage_fname = ''
            self.entry_linkage.setText('Filename')
            # Enable embedding browse button only (entry widget should always be disabled)
            self.browse_embedding.setEnabled(True)
            # Disable linkage widgets
            self.browse_linkage.setEnabled(False)
            self.update_cluster_subwidgets_state(False)
        else:
            # When embedding is unchecked, automatically check linkage
            Imports.checkbox_linkage.setChecked(True)
        self.mark_imports_changed()

    def toggle_linkage(self):
        """Handle linkage checkbox state change"""
        if Imports.checkbox_linkage.isChecked():
            # Uncheck embedding checkbox if linkage is checked
            Imports.checkbox_embedding.setChecked(False)
            # Clear embedding filename when switching to linkage
            Imports.embedding_fname = ''
            self.entry_embedding.setText('Filename')
            self.has_embedding = False
            # Enable linkage browse button only (entry widget should always be disabled)
            self.browse_linkage.setEnabled(True)
            # Disable embedding widgets
            self.browse_embedding.setEnabled(False)
            self.entry_embedding.setText('Filename')
            self.has_embedding = False
            # Disable embedding subwidgets when checkbox is unchecked
            self.label_embedding_method.setEnabled(False)
            self.combo_embedding_method.setEnabled(False)
            self.label_embedding_clusters.setEnabled(False)
            self.spin_embedding_clusters.setEnabled(False)
            self.label_embedding_sort_method.setEnabled(False)
            self.combo_embedding_sort_method.setEnabled(False)
        else:
            # When linkage is unchecked, automatically check embedding
            Imports.checkbox_embedding.setChecked(True)
        self.mark_imports_changed()

    def mark_imports_changed(self):
        """Mark that imports have been changed since last confirmation"""
        if Imports.imports_confirmed:
            Imports.imports_changed = True
            Imports.btn_process_imports.setText('Update Imports')
            Imports.btn_process_imports.setToolTip('Apply changes to imports before proceeding to analysis.')

    def load_mave(self):
        Imports.mave_fname = QFileDialog.getOpenFileName(self, 'Choose data file', '', ('Data files (*.csv)'))[0]
        if Imports.mave_fname:
            try:
                try:
                    Imports.mave = pd.read_csv(Imports.mave_fname)  # Try comma first
                except:
                    Imports.mave = pd.read_csv(Imports.mave_fname, sep='\t')  # Try tab
                Imports.ref_full = Imports.mave['Sequence'][0]
                self.entry_mave.setDisabled(False)
                self.entry_mave.setText(Imports.mave_fname)
                self.entry_mave.setDisabled(True)
                
                # Set up sequence length controls based on MAVE sequence length
                Imports.startbox_cropped.setMaximum(len(Imports.mave['Sequence'][0])-1)
                Imports.startbox_cropped.setValue(0)
                Imports.stopbox_cropped.setMaximum(len(Imports.mave['Sequence'][0]))
                Imports.stopbox_cropped.setValue(len(Imports.mave['Sequence'][0]))
                Imports.stopbox_cropped.setSuffix(' / %s' % len(Imports.mave['Sequence'][0]))
                
                Imports.combo_ref.setDisabled(False)
                Imports.combo_ref.setCurrentIndex(0)
                Imports.entry_ref.setEnabled(False)
                Imports.entry_ref.setText(Imports.ref_full)
                Imports.entry_ref.setReadOnly(True)
                # Enable MAVE subwidgets
                self.update_mave_subwidgets_state(True)
                tabs.setTabEnabled(1, False)
                tabs.setTabEnabled(2, False)
                Imports.btn_process_imports.setDisabled(False)
                self.mark_imports_changed()
                
            except Exception as e:
                QMessageBox.warning(self, 'Error', str(e))
                return
        else:
            # User cancelled - reset MAVE-related fields and disable subwidgets
            self.entry_mave.setDisabled(False)
            self.entry_mave.setText('Filename')
            self.entry_mave.setDisabled(True)
            # Reset reference sequence fields
            Imports.entry_ref.setText('')
            Imports.combo_ref.setCurrentIndex(0)
            # Disable MAVE subwidgets (this will also reset start/stop positions)
            self.update_mave_subwidgets_state(False)            
            self.mark_imports_changed()

    def select_ref(self):
        if Imports.combo_ref.currentText() == 'First row':
            Imports.ref_full = Imports.mave['Sequence'][0]
            Imports.entry_ref.setEnabled(False)
            Imports.entry_ref.setText(Imports.ref_full)
        elif Imports.combo_ref.currentText() == 'Custom':
            Imports.combo_ref.setDisabled(False)
            Imports.entry_ref.setText('')
            Imports.entry_ref.setEnabled(True)
        elif Imports.combo_ref.currentText() == 'None':
            Imports.entry_ref.setText('')
            Imports.entry_ref.setEnabled(False)
            Imports.ref_full = ''
        try:
            tabs.setTabEnabled(1, False)
            tabs.setTabEnabled(2, False)
            Imports.btn_process_imports.setDisabled(False)
        except:
            pass
        self.mark_imports_changed()

    def load_maps(self):
        Imports.maps_fname = QFileDialog.getOpenFileName(self, 'Choose data file', '', ('Data files (*.npy)'))[0]
        if Imports.maps_fname:
            self.entry_maps.setDisabled(False)
            self.entry_maps.setText(Imports.maps_fname)
            self.entry_maps.setDisabled(True)
            
            # Load and process the maps file
            try:
                Imports.maps = np.load(Imports.maps_fname, allow_pickle=True)
                Imports.maps_shape = Imports.maps.shape
                Imports.nS = Imports.maps_shape[0]
                Imports.seq_length = Imports.maps_shape[1]
                
                # Enable maps subwidgets
                self.update_maps_subwidgets_state(True)
                
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Failed to load maps file: {str(e)}')
                return
        else:
            # User cancelled - disable maps subwidgets
            self.entry_maps.setDisabled(False)
            self.entry_maps.setText('Filename')
            self.entry_maps.setDisabled(True)
            self.update_maps_subwidgets_state(False)        
        self.mark_imports_changed()

    def apply_linkage_cut(self): # TODO why is this happening on P1?
        criterion = self.combo_cut_criterion.currentText()
        param = self.spin_cut_param.value()
        if not hasattr(Imports, 'maps') or Imports.maps is None:
            QMessageBox.warning(self, 'Error', 'Please load attribution maps before cutting linkage.')
            return
        clusterer = Clusterer(Imports.maps.reshape((Imports.nS, Imports.seq_length, Imports.maps_shape[2])), method='umap', gpu=True)
        if criterion == 'maxclust':
            labels, _ = clusterer.get_cluster_labels(Imports.linkage, criterion='maxclust', n_clusters=param)
            self.label_cut_param.setText('Number of clusters:')
        else:
            labels, _ = clusterer.get_cluster_labels(Imports.linkage, criterion='distance', max_distance=param)
            self.label_cut_param.setText('Distance threshold:')
        Imports.clusters = labels

    def cropped_signal(self):
        Imports.btn_process_imports.setDisabled(False)
        self.mark_imports_changed()

    def load_embedding(self):
        Imports.embedding_fname = QFileDialog.getOpenFileName(self, 'Choose data file', '', ('Data files (*.npy)'))[0]
        if Imports.embedding_fname:
            self.entry_embedding.setDisabled(False)
            self.entry_embedding.setText(Imports.embedding_fname)
            self.entry_embedding.setDisabled(True)
            
            # Enable embedding subwidgets when file is loaded
            self.label_embedding_method.setEnabled(True)
            self.combo_embedding_method.setEnabled(True)
            self.label_embedding_clusters.setEnabled(True)
            self.spin_embedding_clusters.setEnabled(True)
            self.label_embedding_sort_method.setEnabled(True)
            self.combo_embedding_sort_method.setEnabled(True)
            
            self.has_embedding = True
            
            # Only enable button if both embedding and linkage are loaded
            Imports.btn_process_imports.setDisabled(False)
            fname_contains = 'crop_'
            if Imports.embedding_fname.__contains__('_crop_'):
                try:
                    fname_end = Imports.embedding_fname.split(fname_contains, 1)[1]
                    fname_start, fname_stop = fname_end.split('_', 1)
                    fname_stop = fname_stop.split('.', 1)[0]
                    msg = "<span style='font-weight:normal;'>\
                        Map start and stop positions recognized in filename.\
                        <br /><br />\
                        Fill in the corresponding entries above?\
                        </span>"
                    box = QMessageBox(self)
                    box.setWindowTitle('%s Inputs' % progname)
                    box.setText('<b>Inputs Query</b>')
                    box.setFont(font_standard)
                    box.setIcon(QMessageBox.Question)
                    box.setInformativeText(msg)
                    box.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
                    reply = box.exec_()
                    if reply == QMessageBox.Yes:
                        Imports.startbox_cropped.setValue(int(fname_start))
                        Imports.stopbox_cropped.setValue(int(fname_stop))
                except:
                    pass
        else:
            # User cancelled - reset entry widget and disable subwidgets
            self.entry_embedding.setDisabled(False)
            self.entry_embedding.setText('Filename')
            self.entry_embedding.setDisabled(True)
            
            # Disable embedding subwidgets when browse is cancelled
            self.label_embedding_method.setEnabled(False)
            self.combo_embedding_method.setEnabled(False)
            self.label_embedding_clusters.setEnabled(False)
            self.spin_embedding_clusters.setEnabled(False)
            self.label_embedding_sort_method.setEnabled(False)
            self.combo_embedding_sort_method.setEnabled(False)
            
            self.has_embedding = False
            Imports.btn_process_imports.setDisabled(False)
        
        tabs.setTabEnabled(1, False)
        tabs.setTabEnabled(2, False)        
        self.mark_imports_changed()

    def load_clusters_or_linkage(self):
        Imports.linkage_fname = QFileDialog.getOpenFileName(self, 'Choose linkage file', '', 'All files (*.npy)')[0]
        if not Imports.linkage_fname:
            # User cancelled - reset entry widget
            self.entry_linkage.setDisabled(False)
            self.entry_linkage.setText('Filename')
            self.entry_linkage.setDisabled(True)
            self.has_clusters = False
            self.update_cluster_subwidgets_state(False)
            Imports.btn_process_imports.setDisabled(True)
            self.mark_imports_changed()
            return
        
        self.entry_linkage.setDisabled(False)
        self.entry_linkage.setText(Imports.linkage_fname)
        self.entry_linkage.setDisabled(True)
        
        arr = np.load(Imports.linkage_fname, allow_pickle=True)
        
        # Check if array is linkage matrix (2D array with 4 columns)
        if arr.ndim == 2 and arr.shape[1] == 4:
            Imports.linkage = arr
            self.has_clusters = True
            # Enable cluster subwidgets for linkage matrix
            self.update_cluster_subwidgets_state(True)
        else:
            # Not a valid linkage matrix - show error and reset
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setText('<b>Invalid Linkage File</b>')
            box.setFont(font_standard)
            box.setStandardButtons(QMessageBox.Ok)
            box.setInformativeText('Linkage file must contain a 2D array with 4 columns (linkage matrix).\nShape should be (N-1, 4) where N is the number of sequences.')
            reply = box.exec_()
            
            # Reset entry widget
            self.entry_linkage.setDisabled(False)
            self.entry_linkage.setText('Filename')
            self.entry_linkage.setDisabled(True)
            self.has_clusters = False
            self.update_cluster_subwidgets_state(False)
            Imports.btn_process_imports.setDisabled(True)
            Imports.linkage_fname = ''
            Imports.linkage = None

        # Mark imports as changed
        self.mark_imports_changed()


################################################################################
# Custom clustering tab 

class Custom(QtWidgets.QWidget):
    # for changing views based on embedding coordinates chosen:
    eig1_choice = 0 
    eig2_choice = 1
    coordsX = [] # user X coordinate picks
    coordsY = [] # user Y coordinate picks
    connected = 0 # binary : 0=unconnected, 1=connected
    pts_orig = []
    pts_origX = []
    pts_origY = []
    pts_new = []
    pts_newX = []
    pts_newY = []
    pts_encircled = []
    pts_encircledX = []
    pts_encircledY = []
    x = []
    y = []
    imgAvg = []
    plt_step = 1 # for n>1, every (n-1)th point will be skipped when rendering scatter plots
    plt_marker = .5
    plt_lw = 0
    plt_zorder = 'Original'
    zord = []
    zord0 = []
    theme_mode = 'Light   '
    theme_color1 = 'black'
    theme_color2 = 'red'
    cmap = 'DNN'
    cbar_label = 'DNN score'
    idx_encircled = []
    logo_label = 'Enrichment   '
    logo_ref = False
    contributions = False # input x gradient mode
    logos_start = 0
    logos_stop = 1
    plot_ref = False
    clusters = '' # TODO - remove and set up properly

    def __init__(self, parent=None):
        super(Custom, self).__init__(parent)

        Custom.figure = Figure(dpi=dpi)
        Custom.ax = self.figure.add_subplot(111)
        Custom.ax.set_aspect('equal')
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        def choose_eig1():
            Custom.eig1_choice = int(Custom.entry_eig1.value()-1)
            Custom.btn_reset.click()
        
        def choose_eig2():
            Custom.eig2_choice = int(Custom.entry_eig2.value()-1)
            Custom.btn_reset.click()

        def choose_stepsize():
            Custom.plt_step = Custom.entry_stepsize.value() + 1
            Custom.btn_reset.click()
            if Imports.clusters_dir != '': # need to also update tab 3: 
                Predef.cluster_colors = Imports.clusters[Imports.cluster_col][::Custom.plt_step]

        def choose_markersize():
            Custom.plt_marker = Custom.entry_markersize.value()
            Custom.btn_reset.click()

        def choose_cmap():
            Custom.cmap = Custom.entry_cmap.currentText()
            #if Custom.entry_cmap.currentText() == 'None':
                #Custom.cbar_label = 'n/a'
            if Custom.entry_cmap.currentText() == 'DNN':
                Custom.cbar_label = 'DNN score'
            elif Custom.entry_cmap.currentText() == 'GIA':
                Custom.cbar_label = 'GIA score'
            elif Custom.entry_cmap.currentText() == 'Hamming':
                Custom.cbar_label = 'Hamming distance'
            elif Custom.entry_cmap.currentText() == 'Task':
                Custom.cbar_label = 'DNN task'
            elif Custom.entry_cmap.currentText() == 'Histogram':
                Custom.cbar_label = 'Histogram'
            
            # Update zorder based on current drawing order setting and new color map
            if Custom.entry_zorder.currentText() == 'Original':
                Custom.zord = Custom.zord0
            elif Custom.entry_zorder.currentText() == 'Ascending':
                if Custom.cmap == 'Histogram':
                    Custom.zord = Custom.zord0
                else:
                    Custom.zord = np.argsort(np.array(Imports.mave[Custom.cmap]))
            elif Custom.entry_zorder.currentText() == 'Descending':
                if Custom.cmap == 'Histogram':
                    Custom.zord = Custom.zord0
                else:
                    Custom.zord = np.argsort(np.array(Imports.mave[Custom.cmap]))[::-1]
            
            Custom.btn_reset.click()

        def choose_zorder():
            Custom.plt_zorder = Custom.entry_zorder.currentText()
            if Custom.entry_zorder.currentText() == 'Original':
                Custom.zord = Custom.zord0
            elif Custom.entry_zorder.currentText() == 'Ascending':
                if Custom.cmap == 'Histogram':
                    Custom.zord = Custom.zord0
                else:
                    Custom.zord = np.argsort(np.array(Imports.mave[Custom.cmap]))
            elif Custom.entry_zorder.currentText() == 'Descending':
                if Custom.cmap == 'Histogram':
                    Custom.zord = Custom.zord0
                else:
                    Custom.zord = np.argsort(np.array(Imports.mave[Custom.cmap]))[::-1]
            Custom.btn_reset.click()

        def choose_theme():
            Custom.theme_mode = Custom.entry_theme.currentText()
            if Custom.theme_mode == 'Light   ':
                Custom.theme_color1 = 'black'
                Custom.theme_color2 = 'red'
            elif Custom.theme_mode == 'Dark   ':
                Custom.theme_color1 = 'lime'
                Custom.theme_color2 = 'cyan'
            Custom.btn_reset.click()

        def plot_reference():
            if Imports.ref_idx != '': # TODO - add warning or disable
                if Custom.checkbox_ref.isChecked():
                    Custom.plot_ref = True 
                else:
                    Custom.plot_ref = False
                self.reset()
            else:
                Custom.checkbox_ref.setChecked(False)
                Custom.checkbox_ref.setDisabled(True)
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Warning)
                box.setText('<b>Input Warning</b>')
                box.setFont(font_standard)
                if Imports.combo_ref.currentText() == 'None':
                    msg = 'No reference sequence provided on imports tab.'
                elif Imports.combo_ref.currentText() == 'Custom':
                    msg = 'No match to chosen reference sequence found in MAVE dataset.'
                box.setStandardButtons(QMessageBox.Ok)
                box.setInformativeText(msg)
                reply = box.exec_()

        #for tick in Custom.ax.xaxis.get_major_ticks():
        #    tick.label.set_fontsize(4)
        #for tick in Custom.ax.yaxis.get_major_ticks():
        #    tick.label.set_fontsize(4)
        Custom.ax.tick_params(axis='both', which='major', labelsize=10)
        Custom.ax.get_xaxis().set_ticks([])
        Custom.ax.get_yaxis().set_ticks([])
        Custom.ax.set_title('Place points on the plot to encircle cluster(s)', fontsize=5)
        Custom.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (Custom.eig1_choice+1), fontsize=6)
        Custom.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (Custom.eig2_choice+1), fontsize=6)
        Custom.ax.autoscale()
        Custom.ax.margins(x=0)
        Custom.ax.margins(y=0)
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.draw() # refresh canvas

        # canvas widgets:
        Custom.entry_eig1 = QSpinBox(self)
        Custom.entry_eig1.setMinimum(1)
        Custom.entry_eig1.valueChanged.connect(choose_eig1)
        Custom.entry_eig1.setPrefix('%s1: ' % u"\u03A8")
        Custom.entry_eig1.setToolTip('Select the eigenvector to display on the first (2D) coordinate.')
        Custom.entry_eig1.setAlignment(QtCore.Qt.AlignCenter)

        Custom.entry_eig2 = QSpinBox(self)
        Custom.entry_eig2.setMinimum(2)
        Custom.entry_eig2.valueChanged.connect(choose_eig2)
        Custom.entry_eig2.setPrefix('%s2: ' % u"\u03A8")
        Custom.entry_eig2.setToolTip('Select the eigenvector to display on the second (2D) coordinate.')
        Custom.entry_eig2.setAlignment(QtCore.Qt.AlignCenter)

        Custom.btn_reset = QPushButton('Reset Path')
        Custom.btn_reset.clicked.connect(self.reset)
        Custom.btn_reset.setDisabled(False)
        Custom.btn_reset.setDefault(False)
        Custom.btn_reset.setAutoDefault(False)

        self.btn_connect = QPushButton('Connect Path')
        self.btn_connect.clicked.connect(self.connect)
        self.btn_connect.setDisabled(True)
        self.btn_connect.setDefault(False)
        self.btn_connect.setAutoDefault(False)

        self.btn_view = QPushButton('View Cluster')
        self.btn_view.clicked.connect(self.view)
        self.btn_view.setDisabled(True)
        self.btn_view.setDefault(False)
        self.btn_view.setAutoDefault(False)

        self.label_cmap = QLabel('Color map:')
        Custom.entry_cmap = QComboBox(self)
        #Custom.entry_cmap.addItem('None')
        Custom.entry_cmap.addItem('DNN')
        Custom.entry_cmap.addItem('GIA')
        Custom.entry_cmap.addItem('Hamming')
        Custom.entry_cmap.addItem('Task')
        Custom.entry_cmap.addItem('Histogram')
        Custom.entry_cmap.currentTextChanged.connect(choose_cmap)
        Custom.entry_cmap.setToolTip('Select the color mapping for the scatter plot.')
        Custom.entry_cmap.setFocus(True)

        self.label_zorder = QLabel('Drawing order:')
        Custom.entry_zorder = QComboBox(self)
        Custom.entry_zorder.addItem('Original')
        Custom.entry_zorder.addItem('Ascending')
        Custom.entry_zorder.addItem('Descending')
        Custom.entry_zorder.currentTextChanged.connect(choose_zorder)
        Custom.entry_zorder.setToolTip('Select the draw order of points in the scatter plot.')

        self.label_stepsize = QLabel('Skip every:')
        Custom.entry_stepsize = QSpinBox(self)
        Custom.entry_stepsize.setMinimum(0)
        Custom.entry_stepsize.valueChanged.connect(choose_stepsize)
        Custom.entry_stepsize.setToolTip('Define the number of points displayed in the scatter plot.')
        Custom.entry_stepsize.setSuffix(' point(s)')
        Custom.entry_stepsize.setAlignment(QtCore.Qt.AlignCenter)

        self.label_markersize = QLabel('Marker size:')
        Custom.entry_markersize = QDoubleSpinBox(self)
        Custom.entry_markersize.setMinimum(.01)
        Custom.entry_markersize.setMaximum(20)
        Custom.entry_markersize.setValue(.5)
        Custom.entry_markersize.setSingleStep(.1)
        Custom.entry_markersize.setDecimals(2)
        Custom.entry_markersize.valueChanged.connect(choose_markersize)
        Custom.entry_markersize.setToolTip('Define the marker size of points in the scatter plot.')
        Custom.entry_markersize.setAlignment(QtCore.Qt.AlignCenter)

        self.label_theme = QLabel('Theme:')
        Custom.entry_theme = QComboBox(self)
        Custom.entry_theme.addItem('Light   ')
        Custom.entry_theme.addItem('Dark   ')
        Custom.entry_theme.currentTextChanged.connect(choose_theme)
        Custom.entry_theme.setToolTip('Select the theme mode for the scatter plot.')

        self.label_ref = QLabel('Plot reference:')
        Custom.checkbox_ref = QCheckBox(self)
        Custom.checkbox_ref.stateChanged.connect(plot_reference)
        Custom.checkbox_ref.setToolTip('Check to show the reference sequence on plot.')

        self.line_L = QLabel('') # aesthetic line left
        self.line_L.setFont(font_standard)
        self.line_L.setMargin(0)
        self.line_L.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        self.line_R = QLabel('') # aesthetic line right
        self.line_R.setFont(font_standard)
        self.line_R.setMargin(0)
        self.line_R.setFrameStyle(QFrame.HLine | QFrame.Sunken)

        layout = QGridLayout()
        layout.setSizeConstraint(QGridLayout.SetMinimumSize)
        # Main path and clustering buttons
        grid_top = QGridLayout()
        grid_top.addWidget(self.toolbar, 0, 0, 1, 10)
        grid_top.addWidget(self.canvas, 1, 0, 50, 10)
        grid_top.addWidget(self.line_L, 51, 0, 1, 2, QtCore.Qt.AlignVCenter)
        grid_top.addWidget(Custom.btn_reset, 51, 2, 1, 2)
        grid_top.addWidget(self.btn_connect, 51, 4, 1, 2)
        grid_top.addWidget(self.btn_view, 51, 6, 1, 2)
        grid_top.addWidget(self.line_R, 51, 8, 1, 2, QtCore.Qt.AlignVCenter)
        # Movable divider for user settings
        grid_bot = QGridLayout()
        grid_bot.addWidget(Custom.entry_eig1, 52, 1, 1, 1)
        grid_bot.addWidget(self.label_cmap, 52, 2, 1, 1, QtCore.Qt.AlignRight)
        grid_bot.addWidget(Custom.entry_cmap, 52, 3, 1, 1, QtCore.Qt.AlignLeft)
        grid_bot.addWidget(self.label_zorder, 52, 4, 1, 1, QtCore.Qt.AlignRight)
        grid_bot.addWidget(Custom.entry_zorder, 52, 5, 1, 1, QtCore.Qt.AlignLeft)
        grid_bot.addWidget(self.label_stepsize, 52, 6, 1, 1, QtCore.Qt.AlignRight)
        grid_bot.addWidget(Custom.entry_stepsize, 52, 7, 1, 1, QtCore.Qt.AlignLeft)
        grid_bot.addWidget(Custom.entry_eig2, 53, 1, 1, 1)
        grid_bot.addWidget(self.label_theme, 53, 2, 1, 1, QtCore.Qt.AlignRight)
        grid_bot.addWidget(Custom.entry_theme, 53, 3, 1, 1, QtCore.Qt.AlignLeft)
        grid_bot.addWidget(self.label_ref, 53, 4, 1, 1, QtCore.Qt.AlignRight)
        grid_bot.addWidget(Custom.checkbox_ref, 53, 5, 1, 1, QtCore.Qt.AlignLeft)
        grid_bot.addWidget(self.label_markersize, 53, 6, 1, 1, QtCore.Qt.AlignRight)
        grid_bot.addWidget(Custom.entry_markersize, 53, 7, 1, 1, QtCore.Qt.AlignLeft)

        widget_top = QWidget()
        widget_top.setLayout(grid_top)
        widget_bot = QWidget()
        widget_bot.setLayout(grid_bot)
        self.splitter = QSplitter(QtCore.Qt.Vertical) # movable divider
        self.splitter.addWidget(widget_top)
        self.splitter.addWidget(widget_bot)
        self.splitter.setStretchFactor(1,1)
        layout.addWidget(self.splitter)
        self.setLayout(layout)


    def reset(self):
        Custom.entry_eig1.setDisabled(False)
        Custom.entry_eig2.setDisabled(False)
        Custom.entry_cmap.setDisabled(False)
        Custom.entry_theme.setDisabled(False)
        Custom.entry_zorder.setDisabled(False)
        Custom.entry_stepsize.setDisabled(False)
        Custom.entry_markersize.setDisabled(False)
        Custom.checkbox_ref.setDisabled(False)

        x = Imports.embedding[:,Custom.eig1_choice]
        y = Imports.embedding[:,Custom.eig2_choice]
        Custom.pts_orig = zip(x,y)
        Custom.pts_origX = x
        Custom.pts_origY = y

        if len(Custom.ax.lines) != 0:
            if self.connected == 1:
                Custom.ax.lines == 0
                self.connected = 0
        else:
            self.connected = 0
        self.btn_connect.setDisabled(True)
        self.btn_view.setDisabled(True)
        self.coordsX = []
        self.coordsY = []

        # Redraw and resize figure
        Custom.ax.clear()
        Custom.cax.cla()
        if Custom.cmap == 'Histogram':
            Custom.ax.set_title('Place points on the plot to encircle cluster(s)', fontsize=5, color='white')
            H, edges = np.histogramdd(np.vstack((Custom.pts_origX, Custom.pts_origY)).T,
                                      bins=101, range=((0,1),(0,1)), density=False, weights=None)
            Custom.scatter = Custom.ax.pcolormesh(H.T, cmap='viridis')
        else:
            Custom.ax.set_title('Place points on the plot to encircle cluster(s)', fontsize=5)
            if Custom.theme_mode == 'Light   ':
                Custom.scatter = Custom.ax.scatter(Custom.pts_origX[Custom.zord][::Custom.plt_step],
                                                 Custom.pts_origY[Custom.zord][::Custom.plt_step],
                                                 c=Imports.mave[Custom.cmap][Custom.zord][::Custom.plt_step],
                                                 s=Custom.plt_marker, cmap='jet', linewidth=Custom.plt_lw)
                                                 #c='k', s=Custom.plt_marker, linewidth=Custom.plt_lw, alpha=0.15)

                Custom.ax.set_facecolor('white')
            elif Custom.theme_mode == 'Dark   ':
                Custom.scatter = Custom.ax.scatter(Custom.pts_origX[Custom.zord][::Custom.plt_step],
                                                 Custom.pts_origY[Custom.zord][::Custom.plt_step],
                                                 c=Imports.mave[Custom.cmap][Custom.zord][::Custom.plt_step],
                                                 s=Custom.plt_marker, cmap='RdBu', linewidth=Custom.plt_lw) 
                Custom.ax.set_facecolor('dimgray')

            if Custom.plot_ref == True:
                Custom.ax.scatter(Custom.pts_origX[Imports.ref_idx], Custom.pts_origY[Imports.ref_idx], marker='*', c='k', zorder=100)

        Custom.ax.tick_params(axis='both', which='major', labelsize=4)
        Custom.ax.get_xaxis().set_ticks([])
        Custom.ax.get_yaxis().set_ticks([])
        Custom.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (Custom.eig1_choice+1), fontsize=6)
        Custom.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (Custom.eig2_choice+1), fontsize=6)
        Custom.ax.autoscale()
        Custom.ax.margins(x=0)
        Custom.ax.margins(y=0)
        
        # Set thin spine thickness for embedding plot
        for spine in Custom.ax.spines.values():
            spine.set_linewidth(0.1)
        
        # Update colorbar
        Custom.cbar.update_normal(Custom.scatter)
        Custom.cbar.ax.set_ylabel(Custom.cbar_label, rotation=270, fontsize=6, labelpad=9)
        Custom.cbar.ax.tick_params(labelsize=6)
        self.canvas.draw()
        

    def connect(self):
        Custom.entry_eig1.setDisabled(True)
        Custom.entry_eig2.setDisabled(True)
        Custom.entry_cmap.setDisabled(True)
        Custom.entry_theme.setDisabled(True)
        Custom.entry_zorder.setDisabled(True)
        Custom.entry_stepsize.setDisabled(True)
        Custom.entry_markersize.setDisabled(True)
        Custom.checkbox_ref.setDisabled(True)

        if len(self.coordsX) > 2:
            # Hack to get better sensitivity of readouts from polygon
            self.coordsX.append(self.coordsX[-1])
            self.coordsY.append(self.coordsY[-1])

            Custom.pts_encircled = []
            Custom.pts_encircledX = []
            Custom.pts_encircledY = []
            codes = []
            for i in range(len(self.coordsX)):
                if i == 0:
                    codes.extend([pltPath.Path.MOVETO])
                elif i == len(self.coordsX)-1:
                    codes.extend([pltPath.Path.CLOSEPOLY])
                else:
                    codes.extend([pltPath.Path.LINETO])
            
            path = pltPath.Path(list(map(list, zip(self.coordsX, self.coordsY))), codes)
            inside = path.contains_points(np.dstack((Custom.pts_origX, Custom.pts_origY))[0].tolist(), radius=1e-9)
            Custom.idx_encircled = []
            index = 0
            Custom.index_enc = 0
            for i in inside:
                index += 1
                if i == True:
                    Custom.index_enc += 1
                    Custom.pts_encircledX.append(Custom.pts_origX[index-1])
                    Custom.pts_encircledY.append(Custom.pts_origY[index-1])
                    Custom.pts_encircled = zip(Custom.pts_encircledX, Custom.pts_encircledY)
                    Custom.idx_encircled.append(index-1)

            # Redraw and resize figure
            Custom.scatter.remove()
            if Custom.cmap == 'Histogram':
                H, edges = np.histogramdd(np.vstack((Custom.pts_origX, Custom.pts_origY)).T,
                                        bins=101, range=((0,1),(0,1)), density=False, weights=None)
                Custom.ax.pcolormesh(H.T, cmap='viridis')
            else:
                Custom.ax.scatter(Custom.pts_origX[::Custom.plt_step], Custom.pts_origY[::Custom.plt_step], s=Custom.plt_marker, c='lightgray', zorder=-100, linewidth=Custom.plt_lw)
                Custom.ax.scatter(Custom.pts_encircledX[::Custom.plt_step], Custom.pts_encircledY[::Custom.plt_step], s=Custom.plt_marker, c=Custom.theme_color2, zorder=-90, linewidth=Custom.plt_lw)
            
            ax = self.figure.axes[0]
            ax.plot([self.coordsX[0],self.coordsX[-1]],
                    [self.coordsY[0],self.coordsY[-1]],
                    color=Custom.theme_color1, linestyle='solid', linewidth=.5, zorder=1)

            # Set thin spine thickness for embedding plot
            for spine in Custom.ax.spines.values():
                spine.set_linewidth(0.1)

            # Update colorbar
            Custom.cbar.update_normal(Custom.scatter)
            Custom.cbar.ax.set_ylabel(Custom.cbar_label, rotation=270, fontsize=6, labelpad=9)
            Custom.cbar.ax.tick_params(labelsize=6)

            self.canvas.draw()
        self.connected = 1
        self.btn_connect.setDisabled(True)
        self.btn_view.setDisabled(False)


    def average_maps():
        seqs = Imports.mave['Sequence'].str.slice(Imports.seq_start, Imports.seq_stop)
        map_avg = np.zeros(shape=(1, Imports.dim))
        bin_count = 0
        idxs = []
        for idx in range(Imports.nS):
            if idx in Custom.idx_encircled:
                if Custom.contributions is False:
                    map_avg += Imports.maps[idx,:]
                else:
                    map_avg += Imports.maps[idx,:] * utils.seq2oh(seqs[idx], Imports.alphabet).flatten()
                bin_count += 1
                idxs.append(idx)
        Custom.seqs_cluster = seqs[idxs]
        if bin_count > 0:
            map_avg /= bin_count
            map_avg = np.reshape(map_avg, (Imports.seq_length, Imports.maps_shape[2]))
            try: # TODO - see below
                Custom.imgAvg = utils.arr2pd(map_avg, Imports.alphabet)
            except: # TODO - may need a separate option for this to deal with two-hot encodings (e.g., CLIPNET)
                Custom.imgAvg = utils.arr2pd(map_avg)
        else:
            Custom.imgAvg = None
        return idxs


    def view(self): # view average of all attribution maps in encircled region
        idxs = Custom.average_maps()   
        if not idxs: # if user selected area on canvas contains no data points
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setText('<b>Input Warning</b>')
            box.setFont(font_standard)
            msg = 'At least one point must be selected.\n\nIf errors occur encircling points, try adding more vertices.'
            box.setStandardButtons(QMessageBox.Ok)
            box.setInformativeText(msg)
            reply = box.exec_()
            Custom.btn_reset.click()
        else:
            # Create matrix based on positional frequencies
            seq_array_cluster = motifs.create(Custom.seqs_cluster, alphabet=Imports.alphabet)
            Custom.pfm_cluster = seq_array_cluster.counts # position frequency matrix
            pseudocounts = 0.5
            # Standard PWM
            pwm_cluster = Custom.pfm_cluster.normalize(pseudocounts=pseudocounts) #https://biopython-tutorial.readthedocs.io/en/latest/notebooks/14%20-%20Sequence%20motif%20analysis%20using%20Bio.motifs.html
            Custom.seq_logo_pwm = pd.DataFrame(pwm_cluster)
            # Enrichment logo (see https://logomaker.readthedocs.io/en/latest/examples.html#ars-enrichment-logo)
            enrichment_ratio = (pd.DataFrame(Custom.pfm_cluster) + pseudocounts) / (pd.DataFrame(Custom.pfm_background) + pseudocounts)
            Custom.seq_logo_enrich = np.log2(enrichment_ratio)
            if Custom.logo_label == 'Enrichment   ':
                Cluster.seq_logo = Custom.seq_logo_enrich
            else:
                Cluster.seq_logo = Custom.seq_logo_pwm
            self.open_cluster_window()

    def open_cluster_window(self):
        global cluster_window
        try:
            cluster_window.close()
        except:
            pass
        cluster_window = Cluster()
        cluster_window.setMinimumSize(10, 10)
        cluster_window.show()

    def onclick(self, event):
        if Custom.cmap == 'Histogram':
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setText('<b>Input Warning</b>')
            box.setFont(font_standard)
            msg = 'Histogram mode enabled.\n\nChange to a different color map to enable canvas interactivity.'
            box.setStandardButtons(QMessageBox.Ok)
            box.setInformativeText(msg)
            reply = box.exec_()
        else:
            if self.connected == 0:
                zooming_panning = (self.figure.canvas.cursor().shape() != 0) # arrow is 0 when not zooming (2) or panning (9)
                if not zooming_panning:
                    ix, iy = event.xdata, event.ydata
                    if ix != None and iy != None:
                        self.coordsX.append(float(ix))
                        self.coordsY.append(float(iy))
                        ax = self.figure.axes[0]
                        ax.plot(event.xdata, event.ydata, color=Custom.theme_color1, marker='.', markersize=2.5, zorder=2)
                        if len(self.coordsX) > 1:
                            x0, y0 = self.coordsX[-2], self.coordsY[-2]
                            x1, y1 = self.coordsX[-1], self.coordsY[-1]
                            ax.plot([x0,x1],[y0,y1], color=Custom.theme_color1, linestyle='solid', linewidth=.5, zorder=1)

                        self.canvas.draw()
                    if len(self.coordsX) > 2:
                        self.btn_connect.setDisabled(False)


# =============================================================================
# Plot average of logos and related statistics within encircled region:
# =============================================================================

class Cluster(QtWidgets.QMainWindow):
    seq_logo = None
    batch_logo_attribution = None  # Store BatchLogo instance for attribution logo
    batch_logo_sequence = None     # Store BatchLogo instance for sequence logo

    def __init__(self):
        super(Cluster, self).__init__()
        self.left = 10
        self.top = 10
        self.figure = Figure(dpi=dpi, constrained_layout=True)
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)

        self.toolbar = NavigationToolbar(self.canvas, self)
        unwanted_buttons = ['Subplots']
        for x in self.toolbar.actions():
            if x.text() in unwanted_buttons:
                self.toolbar.removeAction(x)

        self.updateGeometry()
        centralwidget = QWidget()
        self.setCentralWidget(centralwidget)

        def choose_logostyle():
            if self.entry_logostyle.currentText() == 'Enrichment   ':
                Custom.logo_label = self.entry_logostyle.currentText()
                Cluster.seq_logo = Custom.seq_logo_enrich
            elif self.entry_logostyle.currentText() == 'PWM   ':
                Custom.logo_label = self.entry_logostyle.currentText()
                Cluster.seq_logo = Custom.seq_logo_pwm
            self.reset()

        def show_contributions():
            if Cluster.checkbox_contrib.isChecked():
                Custom.contributions = True
            else:
                Custom.contributions = False
            idxs = Custom.average_maps()
            self.reset()

        def color_reference():
            if Cluster.checkbox_ref.isChecked():
                Custom.logo_ref = True
            else:
                Custom.logo_ref = False
            self.reset()

        def print_sequence_indices():
            print('\nSequence indices:', list(Custom.idx_encircled))

        def open_sequence_table():
            seqs_cluster_plus = []
            for i in range(len(Custom.seqs_cluster)+2):
                if i == 0:
                    seqs_cluster_plus.append(Custom.pfm_background.consensus)
                elif i == 1:
                    seqs_cluster_plus.append(Custom.pfm_cluster.consensus)
                elif i > 1:
                    seqs_cluster_plus.append(Custom.seqs_cluster.values[i-2])
            self.table = SequenceTable(data=seqs_cluster_plus)
            self.table.show()

        def crop_logos():
            if self.entry_start.value() < self.entry_stop.value():
                Custom.logos_start = int(self.entry_start.value())
                Custom.logos_stop = int(self.entry_stop.value())
                self.reset()
            else:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Warning)
                box.setText('<b>Input Warning</b>')
                box.setFont(font_standard)
                msg = 'The start position cannot be greater than the stop position.'
                box.setStandardButtons(QMessageBox.Ok)
                box.setInformativeText(msg)
                reply = box.exec_()

        def open_stats_window(self):
            global cluster_stats_window
            try:
                cluster_stats_window.close()
            except:
                pass
            if Imports.current_tab == 1:  # Custom tab
                cluster_stats_window = Stats(Custom)
            elif Imports.current_tab == 2:  # Predef tab
                # Find the current Predef instance
                for widget in self.parent().findChildren(Predef):
                    if widget.isVisible():
                        cluster_stats_window = Stats(widget)
                        break
            cluster_stats_window.setMinimumSize(10, 10)
            cluster_stats_window.show()

        # Create widgets
        self.label_view = QLabel('Logo window: ')
        self.entry_start = QDoubleSpinBox(self)
        self.entry_start.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.entry_start.setDecimals(0)
        self.entry_start.setMinimum(0)
        self.entry_start.setMaximum(Imports.seq_length-1)
        self.entry_start.setValue(Custom.logos_start)
        self.entry_start.setPrefix('Start: ')
        self.entry_start.setToolTip('Start position for viewing window.')
        self.entry_start.lineEdit().returnPressed.connect(crop_logos)

        self.entry_stop = QDoubleSpinBox(self)
        self.entry_stop.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.entry_stop.setDecimals(0)
        self.entry_stop.setMinimum(1)
        self.entry_stop.setMaximum(Imports.seq_length)
        self.entry_stop.setValue(Custom.logos_stop)
        self.entry_stop.setPrefix('Stop: ')
        self.entry_stop.setSuffix(' / %s' % int(Imports.seq_length))
        self.entry_stop.setToolTip('Stop position for viewing window.')
        self.entry_stop.lineEdit().returnPressed.connect(crop_logos)

        self.btn_crop = QPushButton('Crop view')
        self.btn_crop.clicked.connect(crop_logos)
        self.btn_crop.setDisabled(False)
        self.btn_crop.setDefault(False)
        self.btn_crop.setAutoDefault(False)
        self.btn_crop.setToolTip('Visually crop display window for logos based on start and stop values.')

        self.label_cluster = QLabel('Cluster info: ')
        self.btn_seq_table = QPushButton('Sequences')
        self.btn_seq_table.clicked.connect(open_sequence_table)
        self.btn_seq_table.setDisabled(False)
        self.btn_seq_table.setDefault(False)
        self.btn_seq_table.setAutoDefault(False)
        self.btn_seq_table.setToolTip('View all sequences in the current cluster.')

        self.btn_seq_stats = QPushButton('Statistics')
        self.btn_seq_stats.clicked.connect(open_stats_window)
        self.btn_seq_stats.setDisabled(False)
        self.btn_seq_stats.setDefault(False)
        self.btn_seq_stats.setAutoDefault(False)
        self.btn_seq_stats.setToolTip('View statistics based on sequences in the current cluster.')

        self.label_display = QLabel('Logo display: ')
        Cluster.entry_logostyle = QComboBox(self)
        Cluster.entry_logostyle.addItem('Enrichment   ')
        Cluster.entry_logostyle.addItem('PWM   ')
        if Custom.logo_label == 'Enrichment   ':
            Cluster.entry_logostyle.setCurrentIndex(0)
        elif Custom.logo_label == 'PWM   ':
            Cluster.entry_logostyle.setCurrentIndex(1)
        Cluster.entry_logostyle.currentTextChanged.connect(choose_logostyle)
        Cluster.entry_logostyle.setToolTip('Select the visualization scheme for the sequence statistics.')

        if Custom.contributions == True:
            Cluster.checkbox_contrib.setChecked(True)
        else:
            Cluster.checkbox_contrib.setChecked(False)
        Cluster.checkbox_contrib.stateChanged.connect(show_contributions)
        Cluster.checkbox_contrib.setToolTip('This multiplies each sequence\'s attribution map by its one-hot encoding before averaging.')

        if Custom.logo_ref == True:
            Cluster.checkbox_ref.setChecked(True)
        else:
            Cluster.checkbox_ref.setChecked(False)
        Cluster.checkbox_ref.stateChanged.connect(color_reference)
        Cluster.checkbox_ref.setToolTip('Color logo according to reference sequence.')

        # In Cluster.__init__ (with other widgets):
        Cluster.btn_print_indices = QPushButton('Print sequence indices')
        Cluster.btn_print_indices.setToolTip('Print the indices of the encircled sequences in this cluster to CLI.')
        Cluster.btn_print_indices.clicked.connect(print_sequence_indices)
        Cluster.btn_print_indices.setStyleSheet("padding-left: 16px; padding-right: 16px;")
        Cluster.btn_print_indices.setMinimumHeight(self.btn_seq_table.sizeHint().height())

        layout = QGridLayout(centralwidget)
        layout.setSizeConstraint(QGridLayout.SetMinimumSize) 
        layout.addWidget(self.toolbar, 0, 0, 1, 16)
        layout.addWidget(self.canvas, 1, 0, 50, 16)
        layout.addWidget(self.label_cluster, 51, 0, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.btn_seq_table, 51, 1, 1, 1)
        layout.addWidget(self.btn_seq_stats, 51, 2, 1, 1)
        layout.addWidget(self.label_view, 51, 4, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.entry_start, 51, 5, 1, 1)
        layout.addWidget(self.entry_stop, 51, 6, 1, 1)
        layout.addWidget(self.btn_crop, 51, 7, 1, 1)
        layout.addWidget(self.label_display, 51, 9, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.entry_logostyle, 51, 10, 1, 1, QtCore.Qt.AlignLeft)
        layout.addWidget(Cluster.checkbox_ref, 51, 11, 1, 1, QtCore.Qt.AlignLeft)
        layout.addWidget(Cluster.checkbox_contrib, 51, 12, 1, 1, QtCore.Qt.AlignLeft)
        layout.addWidget(Cluster.btn_print_indices, 51, 13, 1, 3, QtCore.Qt.AlignRight)
        tabs.setTabEnabled(0, False) #freezes out 1st tab
        tabs.setTabEnabled(2, False) #freezes out 3rd tab
        tabs.setTabEnabled(1, False) #freezes out parent tab
        
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        self.resize(int(self.width), int(self.height/2.)) #this changes the height
        self.setWindowTitle('Top: Cluster-averaged attribution maps | Bottom: Positional sequence statistics | %s sequences' % len(Custom.seqs_cluster))
        self.show()
        self.draw()

    def draw(self): # Draw logos for the given cluster
        self.ax1 = self.figure.add_subplot(211)
        
        # Get reference sequence if needed
        ref_seq = None
        if Custom.logo_ref:
            if Imports.map_crop:
                ref_seq = Imports.ref_full[Imports.seq_start:Imports.seq_stop]
            else:
                ref_seq = Imports.ref_full
            ref_seq = ref_seq[Custom.logos_start:Custom.logos_stop]
        
        # Only create new BatchLogo instance if data has changed
        data_changed = (Cluster.batch_logo_attribution is None or 
                       not hasattr(Cluster.batch_logo_attribution, 'values') or 
                       Cluster.batch_logo_attribution.values.shape[1] != len(Custom.imgAvg[Custom.logos_start:Custom.logos_stop]) or
                       not np.array_equal(Cluster.batch_logo_attribution.values[0], 
                                        Custom.imgAvg[Custom.logos_start:Custom.logos_stop].values) or
                       Cluster.batch_logo_attribution.ref_seq != ref_seq)
        
        if data_changed:
            # Create a batch dimension for single logo
            attribution_data = Custom.imgAvg[Custom.logos_start:Custom.logos_stop].values[np.newaxis, :, :]
            Cluster.batch_logo_attribution = BatchLogo(
                attribution_data,
                alphabet=Imports.alphabet,
                figsize=(10, 2.5),
                batch_size=1,
                font_name='Arial Rounded MT Bold',
                fade_below=0.5,
                shade_below=0.5,
                width=0.9,
                center_values=not Custom.contributions,
                ref_seq=ref_seq,
                show_progress=False
            )
            Cluster.batch_logo_attribution.process_all()
            if ref_seq is not None:
                Cluster.batch_logo_attribution.style_glyphs_in_sequence(ref_seq)
        
        # Draw to temporary figure and copy to subplot
        fig1, ax1 = Cluster.batch_logo_attribution.draw_single(
            0,  # Only one logo in batch
            fixed_ylim=False,
            border=False
        )
        
        # Copy collections by creating new ones with the same properties
        for collection in ax1.collections:
            # Get the glyph data from the processed logo
            glyphs = Cluster.batch_logo_attribution.processed_logos[0]['glyphs']
            paths = collection.get_paths()
            colors = [glyph['color'] for glyph in glyphs]
            alphas = [glyph['alpha'] for glyph in glyphs]
            
            # Create new collection with the same properties
            new_collection = matplotlib.collections.PathCollection(
                paths,
                facecolors=colors,
                edgecolors='none',
                linewidths=0,
                alpha=alphas
            )
            self.ax1.add_collection(new_collection)
        
        # Copy axis limits and styling
        self.ax1.set_xlim(ax1.get_xlim())
        self.ax1.set_ylim(ax1.get_ylim())
        self.ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        self.ax1.set_xticks([])
        
        # Set single y-tick at the top for attribution logo
        yticks = ax1.get_yticks()
        if len(yticks) > 0:
            max_y = yticks[-1]
            self.ax1.set_yticks([max_y])
            self.ax1.set_yticklabels([f'{max_y:.2f}'])
            self.ax1.tick_params(axis='y', labelsize=6)
        else:
            self.ax1.set_yticks([])
            self.ax1.set_yticklabels([])
        
        for spine in self.ax1.spines.values():
            spine.set_visible(False)
        plt.close(fig1)  # Close temporary figure
        
        # Handle sequence logo
        self.ax2 = self.figure.add_subplot(212, sharex=self.ax1)
        
        # Initialize highlight_ranges before using it
        highlight_ranges = None
        if Custom.logo_ref and ref_seq is not None:
            # Compare reference sequence with logo
            highlight_positions = []
            for i, (ref_char, logo_chars) in enumerate(zip(ref_seq, 
                                                         Cluster.seq_logo[Custom.logos_start:Custom.logos_stop].values)):
                if ref_char in Imports.alphabet:
                    char_idx = Imports.alphabet.index(ref_char)
                    if logo_chars[char_idx] < 0.5:  # If reference character is not dominant
                        highlight_positions.append(i)
            if highlight_positions:
                highlight_ranges = [highlight_positions]
        
        # Only create new BatchLogo instance if data has changed
        seq_data_changed = (Cluster.batch_logo_sequence is None or 
                          not hasattr(Cluster.batch_logo_sequence, 'values') or 
                          Cluster.batch_logo_sequence.values.shape[1] != len(Cluster.seq_logo[Custom.logos_start:Custom.logos_stop]) or
                          not np.array_equal(Cluster.batch_logo_sequence.values[0], 
                                           Cluster.seq_logo[Custom.logos_start:Custom.logos_stop].values) or
                          Cluster.batch_logo_sequence.ref_seq != ref_seq)
        
        if seq_data_changed:
            # Create a batch dimension for single logo
            sequence_data = Cluster.seq_logo[Custom.logos_start:Custom.logos_stop].values[np.newaxis, :, :]
            Cluster.batch_logo_sequence = BatchLogo(
                sequence_data,
                alphabet=Imports.alphabet,
                figsize=(10, 2.5),
                batch_size=1,
                font_name='Arial Rounded MT Bold',
                fade_below=0.5,
                shade_below=0.5,
                width=0.9,
                center_values=True,
                ref_seq=ref_seq,
                show_progress=False
            )
            Cluster.batch_logo_sequence.process_all()
            if ref_seq is not None:
                Cluster.batch_logo_sequence.style_glyphs_in_sequence(ref_seq)
        
        # Draw to temporary figure and copy to our subplot
        highlight_colors = ['red'] if highlight_ranges else None
        fig2, ax2 = Cluster.batch_logo_sequence.draw_single(
            0,  # Only one logo in batch
            fixed_ylim=False,
            border=False,
            highlight_ranges=highlight_ranges,
            highlight_colors=highlight_colors,
            highlight_alpha=0.3
        )
        
        # Copy collections by creating new ones with the same properties
        for collection in ax2.collections:
            # Get the glyph data from the processed logo
            glyphs = Cluster.batch_logo_sequence.processed_logos[0]['glyphs']
            paths = collection.get_paths()
            colors = [glyph['color'] for glyph in glyphs]
            alphas = [glyph['alpha'] for glyph in glyphs]
            
            # Create new collection with the same properties
            new_collection = matplotlib.collections.PathCollection(
                paths,
                facecolors=colors,
                edgecolors='none',
                linewidths=0,
                alpha=alphas
            )
            self.ax2.add_collection(new_collection)
        
        # Copy axis limits and styling
        self.ax2.set_xlim(ax2.get_xlim())
        self.ax2.set_ylim(ax2.get_ylim())
        self.ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        self.ax2.set_xlabel('Position', fontsize=6, va='top', labelpad=6)
        self.ax2.set_xticks([])
        
        # Set single y-tick at the top for sequence logo
        yticks = ax2.get_yticks()
        if len(yticks) > 0:
            max_y = yticks[-1]
            self.ax2.set_yticks([max_y])
            self.ax2.set_yticklabels([f'{max_y:.2f}'])
            self.ax2.tick_params(axis='y', labelsize=6)
        else:
            self.ax2.set_yticks([])
            self.ax2.set_yticklabels([])
        
        for spine in self.ax2.spines.values():
            spine.set_visible(False)
        plt.close(fig2)  # Close temporary figure
        
        # Set up coordinate tracking for both subplots
        def cluster_format_coord(x, y):
            return f'x={int(x)}, y={y:.3f}'
        
        # Set the format_coord for both axes
        self.ax1.format_coord = cluster_format_coord
        self.ax2.format_coord = cluster_format_coord
        
        self.canvas.draw()

    def reset(self):
        # Clear BatchLogo instances to force recreation
        Cluster.batch_logo_attribution = None
        Cluster.batch_logo_sequence = None
        
        # Clear the figure and redraw
        self.figure.clear()
        self.draw()

    def closeEvent(self, ce): # activated when user clicks to exit via subwindow button
        try:
            self.table.close()
        except:
            pass
        try:
            cluster_stats_window.close()
        except:
            pass
        tabs.setTabEnabled(0, True)
        if Imports.embedding_fname != '' or Imports.linkage_fname != '':
            tabs.setTabEnabled(2, True)
        if Imports.embedding_fname != '':
            tabs.setTabEnabled(1, True)
        else:
            tabs.setTabEnabled(1, False)


class SequenceTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None, *args, **kwds):
        QtWidgets.QTableWidget.__init__(self, parent)
        self.library_values = kwds['data']
        self.seqs_cluster = kwds.get('seqs_cluster', None)
        self.pfm_cluster = kwds.get('pfm_cluster', None)
        self.BuildTable(self.library_values)
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        self.resize(int(self.width), int(self.height/4.))
        
    def AddToTable(self, values):
        for k, v in enumerate(values):
            self.AddItem(k, v)
            
    def AddItem(self, row, data):
        for column, value in enumerate(data):
            item = QtWidgets.QTableWidgetItem(value)
            item = QtWidgets.QTableWidgetItem(str(value))
            self.setItem(row, column, item)

    def BuildTable(self, values):
        self.setSortingEnabled(False)
        self.setRowCount(len(values))
        self.setColumnCount(Imports.seq_length)
        row_labels = []
        for i in range(len(values)):
            if i == 0:
                row_labels.append('Consensus of background')
            elif i == 1:
                row_labels.append('Consensus of cluster')
            elif i > 1:
                if Imports.current_tab == 1:
                    row_labels.append('Sequence %s' % Custom.seqs_cluster.index[i-2])
                elif Imports.current_tab == 2 and self.seqs_cluster is not None:
                    row_labels.append('Sequence %s' % self.seqs_cluster.index[i-2])
                else:
                    row_labels.append('Sequence %s' % (i-2))
        self.setVerticalHeaderLabels(row_labels)
        column_labels = []
        for i in range(Imports.seq_length):
            column_labels.append(str(i)) # start positional index at 0 to match other displays
        self.setHorizontalHeaderLabels(column_labels)
        if Imports.current_tab == 1:
            self.setWindowTitle('Custom cluster | %s sequences' % Custom.index_enc)
        elif Imports.current_tab == 2:
            self.setWindowTitle('Cluster %s | %s sequences' % (Predef.cluster_idx, Predef.num_seqs))
        self.AddToTable(values)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        
    def AddToTable(self, values):
        for k,  v in enumerate(values):
            self.AddItem(k, v)
            
    def AddItem(self, row, data):
        for column, value in enumerate(data):
            item = QtWidgets.QTableWidgetItem(value)
            item = QtWidgets.QTableWidgetItem(str(value))
            self.setItem(row, column, item)


class Stats(QtWidgets.QMainWindow):
    def __init__(self, predef_instance):
        self.predef = predef_instance
        super(Stats, self).__init__()
        self.left = 10
        self.top = 10
        self.figure = Figure(dpi=dpi, constrained_layout=True)
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        unwanted_buttons = ['Subplots']
        for x in self.toolbar.actions():
            if x.text() in unwanted_buttons:
                self.toolbar.removeAction(x)

        self.updateGeometry()
        centralwidget = QWidget()
        self.setCentralWidget(centralwidget)

        layout = QGridLayout(centralwidget)
        layout.setSizeConstraint(QGridLayout.SetMinimumSize) 
        layout.addWidget(self.toolbar, 0, 0, 1, 12)
        layout.addWidget(self.canvas, 1 ,0, 50, 12)

        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        self.resize(int(self.width*(8/10.)), int(self.height*(9/10.)))
        #self.move(self.frameGeometry().topLeft())
        if Imports.current_tab == 1:
            self.setWindowTitle('Statistics for %s sequences' % Custom.index_enc)
        elif Imports.current_tab == 2:
            self.setWindowTitle('Statistics for %s sequences in cluster %s' % (self.predef.num_seqs, self.predef.cluster_idx))
        self.show()

        if Imports.current_tab == 1:
            total = Custom.index_enc # total number of sequences in cluster
        elif Imports.current_tab == 2:
            total = self.predef.num_seqs
        # Matches of sequence positional features in cluster to reference
        self.ax1 = self.figure.add_subplot(231)
        if Imports.combo_ref.currentText() != 'None':
            ref_crop = Imports.ref_full[Imports.seq_start:Imports.seq_stop]
            ref_oh = utils.seq2oh(ref_crop, Imports.alphabet)
            if Imports.current_tab == 1:
                pos_freqs = np.diagonal(np.eye(len(ref_crop))*ref_oh.dot(np.array(pd.DataFrame(Custom.pfm_cluster)).T))
            elif Imports.current_tab == 2:
                pos_freqs = np.diagonal(np.eye(len(ref_crop))*ref_oh.dot(np.array(pd.DataFrame(self.predef.pfm_cluster)).T))
            self.ax1.tick_params(axis="x", labelsize=4)
            self.ax1.tick_params(axis="y", labelsize=4)
            if total > 0:  # Avoid divide by zero
                self.ax1.bar(range(len(ref_crop)), (pos_freqs/total)*100, width=1.0)
            self.ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            self.ax1.get_xaxis().set_ticks([])
            self.ax1.get_yaxis().set_ticks([])
            self.ax1.set_xlim(0,1)
            self.ax1.set_ylim(0,1)
            self.ax1.text(.5, .5, 'N/A', horizontalalignment='center', verticalalignment='center')
        
        # Set title based on tab
        if Imports.current_tab == 1:
            self.ax1.set_title('Custom cluster', fontsize=6)
        elif Imports.current_tab == 2:
            self.ax1.set_title('Cluster %s' % self.predef.cluster_idx, fontsize=6)
        
        self.ax1.set_xlabel('Position', fontsize=4)
        self.ax1.set_ylabel('Reference sequence'
                            '\n'
                            r'matches ($\%$)',
                            fontsize=4)
        # Matches of sequence positional features in cluster to cluster's consensus
        self.ax4 = self.figure.add_subplot(234)
        if Imports.current_tab == 1:
            consensus_seq = Custom.pfm_cluster.consensus
        elif Imports.current_tab == 2: 
            consensus_seq = self.predef.pfm_cluster.consensus
        consensus_oh = utils.seq2oh(consensus_seq, Imports.alphabet)
        if Imports.current_tab == 1:
            pos_freqs = np.diagonal(np.eye(len(consensus_seq))*consensus_oh.dot(np.array(pd.DataFrame(Custom.pfm_cluster)).T))
        elif Imports.current_tab == 2:
            pos_freqs = np.diagonal(np.eye(len(consensus_seq))*consensus_oh.dot(np.array(pd.DataFrame(self.predef.pfm_cluster)).T))
        
        # Set title based on tab
        if Imports.current_tab == 1:
            self.ax4.set_title('Custom cluster', fontsize=6)
        elif Imports.current_tab == 2:
            self.ax4.set_title('Cluster %s' % self.predef.cluster_idx, fontsize=6)
        
        self.ax4.set_xlabel('Position', fontsize=4)
        self.ax4.set_ylabel('Per-cluster consensus'
                            '\n'
                            r'matches ($\%$)',
                            fontsize=4)
        self.ax4.tick_params(axis="x", labelsize=4)
        self.ax4.tick_params(axis="y", labelsize=4)
        if total > 0:  # Avoid divide by zero
            self.ax4.bar(range(len(consensus_seq)), (pos_freqs/total)*100, width=1.0)
        self.ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Distribution of scores in background
        self.ax3 = self.figure.add_subplot(235)
        self.ax3.set_title('Sequences in library', fontsize=6)
        self.ax3.set_xlabel('Activity', fontsize=4)
        self.ax3.set_ylabel('Frequency', fontsize=4)
        self.ax3.tick_params(axis="x", labelsize=4)
        self.ax3.tick_params(axis="y", labelsize=4)
        self.ax3.hist(Imports.mave['DNN'], bins=100)
        if Imports.combo_ref.currentText() == 'First row' or Imports.combo_ref.currentText() == 'Custom':
            try:
                self.ax3.axvline(Imports.mave['DNN'][Imports.ref_idx], c='red', label='Ref.', linewidth=1) # reference prediction
                self.ax3.legend(loc='upper right', fontsize=4, frameon=False)
            except:
                pass
        # Frequency of scores in cluster
        self.ax2 = self.figure.add_subplot(232)
        
        # Set title based on tab
        if Imports.current_tab == 1:
            self.ax2.set_title('Sequences in custom cluster', fontsize=6)
        elif Imports.current_tab == 2:
            self.ax2.set_title('Sequences in cluster %s' % self.predef.cluster_idx, fontsize=6)
        
        self.ax2.set_xlabel('Activity', fontsize=4)
        self.ax2.set_ylabel('Frequency', fontsize=4)
        self.ax2.tick_params(axis="x", labelsize=4)
        self.ax2.tick_params(axis="y", labelsize=4)
        if Imports.current_tab == 1:
            self.ax2.hist(Imports.mave['DNN'][Custom.idx_encircled], bins=100) 
        elif Imports.current_tab == 2:
            self.ax2.hist(Imports.mave['DNN'][self.predef.k_idxs], bins=100)    
        self.ax2.set_xlim(self.ax3.get_xlim())
        
        # Attribution error analysis
        self.ax5 = self.figure.add_subplot(233)
        self.ax5.set_ylabel('Individual attribution map'
                            '\n'
                            'deviation from cluster average', fontsize=4)
        self.ax5.tick_params(axis="x", labelsize=4)
        self.ax5.tick_params(axis="y", labelsize=4)

        if Imports.current_tab == 1:
            maps_cluster = Imports.maps.reshape((Imports.maps.shape[0], Imports.seq_length, Imports.maps_shape[2]))[Custom.idx_encircled]
            errors_cluster = np.linalg.norm(maps_cluster - np.array(Custom.imgAvg), axis=(1,2))
        elif Imports.current_tab == 2:
            maps_cluster = Imports.maps.reshape((Imports.maps.shape[0], Imports.seq_length, Imports.maps_shape[2]))[self.predef.k_idxs]
            map_avg = np.zeros(shape=(1, Imports.dim))
            bin_count = 0
            for idx in range(Imports.nS):
                if idx in self.predef.k_idxs:
                    map_avg += Imports.maps[idx,:]
                    bin_count += 1
            map_avg /= bin_count
            map_avg = np.reshape(map_avg, (Imports.seq_length, Imports.maps_shape[2]))
            try: # TODO - see below
                Predef.imgAvg = utils.arr2pd(map_avg, Imports.alphabet)
            except: # TODO - may need a separate option for this to deal with two-hot encodings (e.g., CLIPNET)
                Predef.imgAvg = utils.arr2pd(map_avg)


            errors_cluster = np.linalg.norm(maps_cluster - np.array(Predef.imgAvg), axis=(1,2))

        if Imports.maps_bg_on is True:
            all_singles = {'Cluster':errors_cluster, 'Background':Imports.errors_background}
            widths = {.45, .45}
        else:
            all_singles = {'Cluster':errors_cluster}
            widths = None
        #flierprops = dict(marker='^', markeredgecolor='k', markerfacecolor='k', linestyle='none', markersize=4)
        boxplot = self.ax5.boxplot(all_singles.values(), showfliers=False, widths=widths,  # default width=0.15
                                   #showmeans=True, meanprops=flierprops,
                                   medianprops={'linestyle': None, 'linewidth': 0}) 
        if Imports.current_tab == 1:
            cluster_label = 'Custom cluster'
        elif Imports.current_tab == 2:
            cluster_label = f'Cluster {self.predef.cluster_idx}'
        
        if Imports.maps_bg_on is True:
            self.ax5.set_xticks([1, 2], [cluster_label, 'Background'], fontsize=5, ha='center')
        else:
            self.ax5.set_xticks([1], [cluster_label], fontsize=5, ha='center')
        for median in boxplot['medians']:
            median.set_color('black')
        pts_cluster = np.random.normal(1, 0.015*3, size=len(errors_cluster)) # 0.015 per 0.15 boxplot width
        self.ax5.scatter(pts_cluster, errors_cluster, s=1, c='C0', linewidth=0, zorder=-10, alpha=0.5)
        if Imports.maps_bg_on is True:
            pts_background = np.random.normal(2, 0.015*3, size=len(Imports.errors_background))
            self.ax5.scatter(pts_background, Imports.errors_background, s=1, c='C0', linewidth=0, zorder=-10, alpha=0.5)
        self.ax5.set_ylim(0, self.ax5.get_ylim()[1])
        self.canvas.draw()


################################################################################
# Predefined clustering tab 

class EmbeddingPlotter:
    """Handles plotting for the embedding pathway (P3)"""
    
    @staticmethod
    def plot_embedding_scatter(ax, embedding, cluster_assignments, cluster_idx, k_idxs, 
                              plt_step, plt_marker, plt_lw, theme_color1, theme_color2):
        """Plot embedding scatter plot with cluster highlighting and colorbar"""
        # Use the exact same logic as the working version for P3
        ax.clear()
        
        # Extract X and Y coordinates from embedding array (like pts_origX and pts_origY)
        pts_origX = embedding[:, 0]
        pts_origY = embedding[:, 1]
        
        # Plot all points with cluster colors (using pts_origX and pts_origY like working version)
        scatter = ax.scatter(pts_origX[::plt_step],
                            pts_origY[::plt_step],
                            s=plt_marker,
                            c=cluster_assignments[::plt_step],
                            cmap='tab10', linewidth=plt_lw, zorder=0)
        
        # Highlight selected cluster points in black (exact same as working version)
        if k_idxs is not None and len(k_idxs) > 0:
            ax.scatter(pts_origX[k_idxs][::plt_step],
                      pts_origY[k_idxs][::plt_step],
                      c='black', 
                      s=plt_marker, cmap='jet', linewidth=plt_lw, zorder=10)
        
        # Set thin spine thickness for embedding plot
        for spine in ax.spines.values():
            spine.set_linewidth(0.1)
            spine.set_visible(True)  # Ensure spines are visible for embedding mode
        
        # Set title
        ax.set_title('Click on a cluster to view its logo', fontsize=5)


class DendrogramPlotter:
    """Handles plotting for the linkage pathway (P3)"""
    
    @staticmethod
    def plot_dendrogram(ax, linkage, maps, spin_cut_param, combo_cut_criterion):
        """Plot dendrogram with proper parameter handling"""
        clusterer = Clusterer(maps, gpu=False)
        
        # Determine cut level for highlighting (exact same as working version)
        param = spin_cut_param.value()
        criterion = combo_cut_criterion.currentText().lower()
        cut_level = None
                
        if criterion == 'maxclust':
            n_clusters = int(param)
            if n_clusters > 1 and n_clusters < Imports.nS:
                # Adjust for 0-based cluster assignments
                cut_level = linkage[Imports.nS - n_clusters - 1, 2]
        else:  # distance
            cut_level = param

        # Always truncate dendrograms (exact same as working version)
        clusterer.plot_dendrogram(linkage, ax=ax, cut_level=cut_level, truncate=True, cut_level_truncate=cut_level, criterion=criterion, n_clusters=param if criterion == 'maxclust' else None, gui=True)
        
        # Set title for linkage mode
        ax.set_title('Dendrogram-based clustering (click on a leaf to view its cluster logo)', fontsize=5)


class Predef(QtWidgets.QWidget):
    pixmap = ''
    coordsX = 0
    coordsY = 0
    cluster_idx = 0  # Initialize as integer 0 instead of string '0'
    num_seqs = 0
    k_idxs = None  # Initialize as None instead of empty string
    logo_dir = 'clusters_avg'
    logo_path = '/y_adaptive'
    clicked = True
    dist = 9001
    df_col = 'Entropy'
    cluster_colors = ''
    cluster_sorted_indices = ''
    clusters_idx = ''
    batch_logo = None  # Store BatchLogo instance for cluster logos
    initial_logo_size = None  # Store initial logo size
    widget_bot_ref = None  # Store reference to widget_bot for splitter sizing
    original_logo_xlim = None  # Store original x-axis limits for home button
    original_logo_ylim = None  # Store original y-axis limits for home button
    show_background_separated = False  # Track background separation checkbox state
    show_variability_logo = False  # Track variability logo checkbox state
    show_average_background = False  # Track average background checkbox state

    def update_logo(self):
        """Update the logo display based on current cluster and settings"""
        if not isinstance(self.cluster_idx, (int, np.integer)) or self.k_idxs is None or not isinstance(self.k_idxs, np.ndarray):
            self.logo_ax.clear()
            self.logo_ax.axis('off')
            self.logo_canvas.draw()
            return

        # Store current view limits if they exist
        current_xlim = None
        current_ylim = None
        had_previous_view = False
        if hasattr(self, 'logo_ax') and self.logo_ax is not None:
            try:
                current_xlim = self.logo_ax.get_xlim()
                current_ylim = self.logo_ax.get_ylim()
                # Check if user had a meaningful x-axis view (not just default 0-1 range)
                if current_xlim[0] != 0 or current_xlim[1] != 1:
                    had_previous_view = True
            except:
                pass

        cluster_id = int(self.cluster_idx)
        logo_type_map = {
            'average of maps': 'average',
            'sequence enrichment': 'enrichment',
            'sequence pwm': 'pwm'
        }
        gui_type = self.entry_logostyle.currentText().lower()
        logo_type = logo_type_map.get(gui_type, 'average')
        
        # Check which logo type is requested
        if Predef.show_variability_logo:
            logo_key = 'variability'
        elif Predef.show_average_background:
            logo_key = 'average_background'
        else:
            # Determine logo key based on background separation setting
            if Predef.show_background_separated:
                logo_key = (cluster_id, f'{logo_type}_separated')
            else:
                logo_key = (cluster_id, logo_type)

        if logo_key not in Imports.batch_logo_instances:
            print(f"Warning: No preprocessed logo found for cluster {cluster_id} with type {logo_type}")
            self.logo_ax.clear()
            self.logo_ax.axis('off')
            self.logo_canvas.draw()
            return

        batch_logo = Imports.batch_logo_instances[logo_key]
        
        # Determine if we should use fixed y-axis limits
        # For variability logo, always use fixed scaling; for others, use widget setting
        if Predef.show_variability_logo:
            use_fixed_ylim = True
        else:
            use_fixed_ylim = self.entry_yscale.currentText() == 'Fixed y-axis'

        # Remove the old logo_canvas from the layout
        parent_layout = self.logo_canvas.parent().layout()
        parent_layout.removeWidget(self.logo_canvas)
        self.logo_canvas.setParent(None)

        # Use consistent figure size to prevent layout shifting
        figsize = (20, 1.5)  # Use a wider, shorter size for better proportions

        # Create figure manually to avoid tight_layout
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw the logo based on type
        if Predef.show_variability_logo:
            # Use pre-computed variability logo figure for instant display
            if hasattr(Imports, 'variability_figure') and hasattr(Imports, 'variability_axis'):
                # Create a fresh copy of the pre-computed figure to avoid size issues
                variability_fig = copy.deepcopy(Imports.variability_figure)
                variability_ax = variability_fig.axes[0]  # Get the axis from the copied figure
                # Close our original figure and use the copied variability figure
                plt.close(fig)
                fig = variability_fig
                ax = variability_ax
            else:
                # Fallback: generate on demand if pre-computed figure not available
                variability_fig, variability_ax = batch_logo.draw_variability_logo(view_window=None, figsize=figsize, border=False)
                plt.close(fig)
                fig = variability_fig
                ax = variability_ax
        elif Predef.show_average_background:
            # Draw average background logo directly (fast, no deep copy needed)
            batch_logo._draw_single_logo(ax, batch_logo.processed_logos[0], fixed_ylim=use_fixed_ylim, border=False)
        else:
            # Draw the logo directly without calling draw_single (which calls tight_layout)
            batch_logo._draw_single_logo(ax, batch_logo.processed_logos[0], fixed_ylim=use_fixed_ylim, border=False)
        
        # Enable tight_layout now that we've removed problematic y-axis labels
        fig.set_tight_layout(True)
        
        # Set y-axis limits based on user selection
        if use_fixed_ylim and not Predef.show_variability_logo:
            # Use pre-calculated global y-axis limits based on background separation setting
            if Predef.show_background_separated and hasattr(Imports, 'global_y_min_separated') and hasattr(Imports, 'global_y_max_separated'):
                # Use background-separated global limits
                ax.set_ylim(Imports.global_y_min_separated, Imports.global_y_max_separated)
                # Set single tick at the top with the value
                ax.set_yticks([Imports.global_y_max_separated])
                ax.set_yticklabels([f'{Imports.global_y_max_separated:.2f}'])
            elif hasattr(Imports, 'global_y_min') and hasattr(Imports, 'global_y_max'):
                # Use standard global limits
                ax.set_ylim(Imports.global_y_min, Imports.global_y_max)
                # Set single tick at the top with the value
                ax.set_yticks([Imports.global_y_max])
                ax.set_yticklabels([f'{Imports.global_y_max:.2f}'])
            else:
                # Fallback to default limits if not calculated
                ax.set_ylim(-1.0, 1.0)
                # Set single tick at the top with the value
                ax.set_yticks([1.0])
                ax.set_yticklabels(['1.00'])
        elif not Predef.show_variability_logo:
            # For adaptive scaling, calculate the max value from the current logo
            yticks = ax.get_yticks()
            if len(yticks) > 0:
                adaptive_max = yticks[-1]
                # Set single tick at the top with the value
                ax.set_yticks([adaptive_max])
                ax.set_yticklabels([f'{adaptive_max:.2f}'])
            else:
                # Fallback if no ticks available
                ax.set_yticks([])
                ax.set_yticklabels([])

        elif Predef.show_average_background:
            # Average background logo uses widget setting for y-axis scaling
            if use_fixed_ylim:
                # Use pre-calculated global y-axis limits
                if hasattr(Imports, 'global_y_min') and hasattr(Imports, 'global_y_max'):
                    ax.set_ylim(Imports.global_y_min, Imports.global_y_max)
                    # Set single tick at the top with the value
                    ax.set_yticks([Imports.global_y_max])
                    ax.set_yticklabels([f'{Imports.global_y_max:.2f}'])
                else:
                    # Fallback to default limits
                    ax.set_ylim(-1.0, 1.0)
                    ax.set_yticks([1.0])
                    ax.set_yticklabels(['1.00'])
            else:
                # For adaptive scaling, calculate the max value from the current logo
                yticks = ax.get_yticks()
                if len(yticks) > 0:
                    adaptive_max = yticks[-1]
                    # Set single tick at the top with the value
                    ax.set_yticks([adaptive_max])
                    ax.set_yticklabels([f'{adaptive_max:.2f}'])
                else:
                    # Fallback if no ticks available
                    ax.set_yticks([])
                    ax.set_yticklabels([])

        elif Predef.show_variability_logo:
            # Variability logo always uses fixed scaling
            if hasattr(Imports, 'global_y_min') and hasattr(Imports, 'global_y_max'):
                ax.set_ylim(Imports.global_y_min, Imports.global_y_max)
                # Set single tick at the top with the value
                ax.set_yticks([Imports.global_y_max])
                ax.set_yticklabels([f'{Imports.global_y_max:.2f}'])
            else:
                # Fallback to default limits
                ax.set_ylim(-1.0, 1.0)
                ax.set_yticks([1.0])
                ax.set_yticklabels(['1.00'])
        
        # Remove y-axis label since we're using tick labels
        if Predef.show_variability_logo:
            ax.set_ylabel('Variability    ')
        elif Predef.show_average_background:
            ax.set_ylabel('Avg. BG    ')
        else:
            ax.set_ylabel('Cluster avg.    ')
        
        # Add coordinate tracking to the logo axis
        def logo_format_coord(x, y):
            return f'x={x:.1f}, y={y:.3f}'
        
        ax.format_coord = logo_format_coord
        
        # Restore previous view limits if they exist
        if had_previous_view and current_xlim is not None:
            ax.set_xlim(current_xlim)
        
        new_canvas = FigureCanvas(fig)
        new_canvas.setMinimumSize(self.logo_canvas.minimumSize())
        new_canvas.setMaximumSize(self.logo_canvas.maximumSize())

        # Create toolbar for the logo canvas
        logo_toolbar = NavigationToolbar(new_canvas, self)
        unwanted_buttons = ['Subplots', 'Zoom', 'Pan']
        for x in logo_toolbar.actions():
            if x.text() in unwanted_buttons:
                logo_toolbar.removeAction(x)

        # Add the new canvas and toolbar to the layout
        parent_layout.addWidget(logo_toolbar, 50, 0, 1, 12)
        parent_layout.addWidget(new_canvas, 51, 0, 1, 12)
        self.logo_canvas = new_canvas
        self.logo_ax = ax

        # Store original view limits for home button functionality
        if Predef.original_logo_xlim is None:
            # First time creating a logo - store the original limits
            Predef.original_logo_xlim = ax.get_xlim()
            Predef.original_logo_ylim = ax.get_ylim()

        # Override the home button action to restore original view
        for action in logo_toolbar.actions():
            if action.text() == 'Home':
                # Disconnect any existing connections to avoid multiple connections
                try:
                    action.triggered.disconnect()
                except:
                    pass
                
                def restore_original_view():
                    ax.set_xlim(Predef.original_logo_xlim)
                    new_canvas.draw()
                action.triggered.connect(restore_original_view)
                break

    def __init__(self, parent=None):
        super(Predef, self).__init__(parent)
        self.logo_fig = Figure(figsize=(8, 1.25), dpi=200)
        self.logo_canvas = FigureCanvas(self.logo_fig)
        self.logo_ax = self.logo_fig.add_subplot(111)
        self.logo_ax.axis('off')

        self.figure = Figure(dpi=dpi)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('equal')
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        def format_coord(x, y):
            #return f'x={x:1.4f}, y={y:1.4f}, cluster={self.cluster_idx:1.0f}, sequences={self.num_seqs:1.0f}'
            return f'cluster={self.cluster_idx:1.0f}, sequences={self.num_seqs:1.0f}'
        
        def cluster_warning():
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setText('<b>Input Warning</b>')
            box.setFont(font_standard)
            msg = 'No cluster selected.'
            box.setStandardButtons(QMessageBox.Ok)
            box.setInformativeText(msg)
            reply = box.exec_()
        
        def open_sequence_table():
            if type(self.k_idxs) != np.ndarray:
                cluster_warning()
            else:
                global predef_table
                seqs_cluster_plus = []
                for i in range(len(self.seqs_cluster)+2):
                    if i == 0:
                        seqs_cluster_plus.append(Custom.pfm_background.consensus)
                    elif i == 1:
                        seqs_cluster_plus.append(self.pfm_cluster.consensus)
                    elif i > 1:
                        seqs_cluster_plus.append(self.seqs_cluster.values[i-2])
                predef_table = SequenceTable(data=seqs_cluster_plus, seqs_cluster=self.seqs_cluster, pfm_cluster=self.pfm_cluster)
                predef_table.show()

        def open_stats_window():
            if type(self.k_idxs) != np.ndarray:
                cluster_warning()
            else:
                global predef_stats_window
                predef_stats_window = Stats(self)
                try:
                    predef_stats_window.close()
                except:
                    pass
                predef_stats_window.setMinimumSize(10, 10)
                predef_stats_window.show()

        def open_all_stats_window():
            if not isinstance(self.df, pd.DataFrame):            
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Warning)
                box.setText('<b>Input Warning</b>')
                box.setFont(font_standard)
                msg = 'No cluster data loaded.'
                box.setStandardButtons(QMessageBox.Ok)
                box.setInformativeText(msg)
                reply = box.exec_()
            else:
                global all_stats_window
                all_stats_window = AllStats()
                try:
                    all_stats_window.close()
                except:
                    pass
                all_stats_window.setMinimumSize(10, 10)
                all_stats_window.show()

        def choose_logostyle():
            """Handle changes in logo style selection"""
            # Map GUI logo types to MetaExplainer canonical types
            logo_type_map = {
                'average of maps': 'average',
                'sequence enrichment': 'enrichment',
                'sequence pwm': 'pwm'
            }
            gui_type = Predef.entry_logostyle.currentText().lower()
            Imports.batch_logo_type = logo_type_map.get(gui_type, 'average')
            
            # The update_logo() function will handle missing logos gracefully
            self.update_logo()

        def choose_yscale():
            """Handle changes in y-axis scale selection"""
            scale_type = 'adaptive' if Predef.entry_yscale.currentText() == 'Adaptive y-axis' else 'fixed'
            if scale_type != Imports.batch_logo_yscale:
                Imports.batch_logo_yscale = scale_type
                # No need to clear cache since logos are pre-generated for both scale types
                self.update_logo()  # Update logo with new scale type

        # Background separation checkbox
        Predef.label_bg_separation = QLabel('Separate background:')
        Predef.label_bg_separation.setFont(font_standard)
        Predef.label_bg_separation.setMargin(20)
        Predef.checkbox_bg_separation = QCheckBox(self)
        Predef.checkbox_bg_separation.setToolTip('Check to show background-separated logos (removes background signal).')
        Predef.checkbox_bg_separation.setChecked(False)
        Predef.checkbox_bg_separation.setVisible(False)  # Hidden by default
        def toggle_background_separation():
            Predef.show_background_separated = Predef.checkbox_bg_separation.isChecked()
            # Uncheck other checkboxes when this one is checked
            if Predef.checkbox_bg_separation.isChecked():
                Predef.checkbox_variability.setChecked(False)
                Predef.checkbox_avg_background.setChecked(False)
                self.update_logo()  # Only update when checking
            else:
                # Only update when unchecking this specific checkbox (not when unchecking due to another checkbox)
                if not Predef.checkbox_variability.isChecked() and not Predef.checkbox_avg_background.isChecked():
                    self.update_logo()
        Predef.checkbox_bg_separation.stateChanged.connect(toggle_background_separation)
        # Store reference to the function for later use
        Predef.toggle_background_separation_func = toggle_background_separation

        # View variability logo checkbox
        Predef.label_variability = QLabel('View variability logo:')
        Predef.label_variability.setFont(font_standard)
        Predef.label_variability.setMargin(20)
        Predef.checkbox_variability = QCheckBox(self)
        Predef.checkbox_variability.setToolTip('Shows a static logo representing sequence variability across all clusters.')
        Predef.checkbox_variability.setChecked(False)
        Predef.checkbox_variability.setVisible(True)  # Always visible
        def toggle_variability():
            Predef.show_variability_logo = Predef.checkbox_variability.isChecked()
            # Uncheck other checkboxes when this one is checked
            if Predef.checkbox_variability.isChecked():
                Predef.checkbox_bg_separation.setChecked(False)
                Predef.checkbox_avg_background.setChecked(False)
                self.update_logo()  # Only update when checking
            else:
                # Only update when unchecking this specific checkbox (not when unchecking due to another checkbox)
                if not Predef.checkbox_bg_separation.isChecked() and not Predef.checkbox_avg_background.isChecked():
                    self.update_logo()
        Predef.checkbox_variability.stateChanged.connect(toggle_variability)
        # Store reference to the function for later use
        Predef.toggle_variability_func = toggle_variability

        # View average background checkbox
        Predef.label_avg_background = QLabel('View average background:')
        Predef.label_avg_background.setFont(font_standard)
        Predef.label_avg_background.setMargin(20)
        Predef.checkbox_avg_background = QCheckBox(self)
        Predef.checkbox_avg_background.setToolTip('Shows a static logo representing the average background signal across all clusters.')
        Predef.checkbox_avg_background.setChecked(False)
        Predef.checkbox_avg_background.setVisible(True)  # Always visible
        def toggle_avg_background():
            Predef.show_average_background = Predef.checkbox_avg_background.isChecked()
            # Uncheck other checkboxes when this one is checked
            if Predef.checkbox_avg_background.isChecked():
                Predef.checkbox_bg_separation.setChecked(False)
                Predef.checkbox_variability.setChecked(False)
                self.update_logo()  # Only update when checking
            else:
                # Only update when unchecking this specific checkbox (not when unchecking due to another checkbox)
                if not Predef.checkbox_bg_separation.isChecked() and not Predef.checkbox_variability.isChecked():
                    self.update_logo()
        Predef.checkbox_avg_background.stateChanged.connect(toggle_avg_background)
        # Store reference to the function for later use
        Predef.toggle_avg_background_func = toggle_avg_background

        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.width = self.canvas.size().width()
        self.ax.format_coord = format_coord

        # Canvas widgets
        Predef.label_pixmap = QLabel()
        # Initialize with blank pixmap instead of trying to load from clusters_dir
        Predef.pixmap = QtGui.QPixmap()
        Predef.pixmap.fill(QtGui.QColor(0,0,0,0))  # Transparent blank pixmap
        Predef.label_pixmap.setScaledContents(True)
        # Set a more reasonable size for the logo display
        Predef.label_pixmap.setMinimumSize(800, 100)  # Minimum size to ensure logo is visible
        Predef.label_pixmap.setMaximumSize(1200, 150)  # Maximum size to prevent excessive scaling
        Predef.label_pixmap.setPixmap(Predef.pixmap)
        Predef.label_pixmap.setAlignment(QtCore.Qt.AlignCenter)  # Center the logo in the label

        self.line_L = QLabel('') # aesthetic line left
        self.line_L.setFont(font_standard)
        self.line_L.setMargin(0)
        self.line_L.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        self.line_R = QLabel('') # aesthetic line right
        self.line_R.setFont(font_standard)
        self.line_R.setMargin(0)
        self.line_R.setFrameStyle(QFrame.HLine | QFrame.Sunken)

        self.btn_seq_table = QPushButton('Clustered Sequences')
        self.btn_seq_table.clicked.connect(open_sequence_table)
        self.btn_seq_table.setDisabled(False)
        self.btn_seq_table.setDefault(False)
        self.btn_seq_table.setAutoDefault(False)
        self.btn_seq_table.setToolTip('View all sequences in the current cluster.')

        self.btn_intra_stats = QPushButton('Intra-cluster Statistics')
        self.btn_intra_stats.clicked.connect(open_stats_window)
        self.btn_intra_stats.setDisabled(False)
        self.btn_intra_stats.setDefault(False)
        self.btn_intra_stats.setAutoDefault(False)
        self.btn_intra_stats.setToolTip('View statistics based on sequences in the current cluster.')

        # Add Cluster activities button
        self.btn_inter_stats = QPushButton('Inter-cluster Statistics')
        self.btn_inter_stats.clicked.connect(self.open_inter_stats_window)
        self.btn_inter_stats.setDisabled(False)
        self.btn_inter_stats.setDefault(False)
        self.btn_inter_stats.setAutoDefault(False)
        self.btn_inter_stats.setToolTip('View boxplot distribution of DNN predictions for all clusters.')
        self.btn_inter_stats.setStyleSheet("padding-left: 16px; padding-right: 16px;")
        self.btn_inter_stats.setMinimumHeight(self.btn_seq_table.sizeHint().height())

        Predef.btn_all_stats = QPushButton('Cluster Summary Matrix (CSM)')
        Predef.btn_all_stats.clicked.connect(open_all_stats_window)
        Predef.btn_all_stats.setDisabled(False)
        Predef.btn_all_stats.setDefault(False)
        Predef.btn_all_stats.setAutoDefault(False)
        Predef.btn_all_stats.setToolTip('View statistics for sequences over all clusters.')

        self.label_choose_cluster = QLabel('Choose cluster: ')
        Predef.choose_cluster = QDoubleSpinBox(self)
        Predef.choose_cluster.setDecimals(0)
        Predef.choose_cluster.setMinimum(0)
        Predef.choose_cluster.setFont(font_standard)
        Predef.choose_cluster.setDisabled(False)
        Predef.choose_cluster.valueChanged.connect(self.highlight_cluster)
        Predef.choose_cluster.setToolTip('Define the index of a cluster of interest.')

        self.label_display = QLabel('Logo display: ')
        Predef.entry_logostyle = QComboBox(self)
        Predef.entry_logostyle.addItem('Average of maps')
        Predef.entry_logostyle.addItem('Sequence enrichment')
        Predef.entry_logostyle.addItem('Sequence PWM')
        Predef.entry_logostyle.currentTextChanged.connect(choose_logostyle)
        Predef.entry_logostyle.setToolTip('Select the logo visualization scheme: Average of maps (attribution-based) or sequence-based options.')
        
        # Disable sequence-based logo options for now
        Predef.entry_logostyle.setItemData(1, False, QtCore.Qt.UserRole - 1)  # Disable 'Sequence enrichment' TODO
        Predef.entry_logostyle.setItemData(2, False, QtCore.Qt.UserRole - 1)  # Disable 'Sequence PWM' TODO

        Predef.entry_yscale = QComboBox(self)
        Predef.entry_yscale.addItem('Adaptive y-axis')
        Predef.entry_yscale.addItem('Fixed y-axis')
        Predef.entry_yscale.currentTextChanged.connect(choose_yscale)
        Predef.entry_yscale.setToolTip('Select the y-axis scaling used for rendering logos.')

        # Truncate dendrogram widgets removed - always truncate by default

        layout = QGridLayout()
        layout.setSizeConstraint(QGridLayout.SetMinimumSize)
        grid_top = QGridLayout()
        grid_top.addWidget(self.toolbar, 0, 0, 1, 12)
        grid_top.addWidget(self.canvas, 1, 0, 50, 12)
        grid_top.addWidget(self.logo_canvas, 51, 0, 1, 12, QtCore.Qt.AlignCenter)
        grid_top.addWidget(self.line_L, 52, 0, 1, 2, QtCore.Qt.AlignVCenter)
        grid_top.addWidget(self.btn_seq_table, 52, 2, 1, 2)
        grid_top.addWidget(self.btn_intra_stats, 52, 4, 1, 2)
        grid_top.addWidget(self.btn_inter_stats, 52, 6, 1, 2)
        grid_top.addWidget(Predef.btn_all_stats, 52, 8, 1, 2)
        grid_top.addWidget(self.line_R, 52, 10, 1, 2, QtCore.Qt.AlignVCenter)
        grid_bot = QGridLayout()
        grid_bot.addWidget(self.label_choose_cluster, 0, 0, 1, 1, QtCore.Qt.AlignRight)
        grid_bot.addWidget(Predef.choose_cluster, 0, 1, 1, 1, QtCore.Qt.AlignLeft)
        grid_bot.addWidget(QVLine(), 0, 3, 1, 2, QtCore.Qt.AlignCenter)
        grid_bot.addWidget(Predef.label_variability, 0, 4, 1, 1, QtCore.Qt.AlignRight)
        grid_bot.addWidget(Predef.checkbox_variability, 0, 5, 1, 1, QtCore.Qt.AlignLeft)
        grid_bot.addWidget(Predef.label_avg_background, 0, 6, 1, 1, QtCore.Qt.AlignRight)
        grid_bot.addWidget(Predef.checkbox_avg_background, 0, 7, 1, 1, QtCore.Qt.AlignLeft)
        grid_bot.addWidget(Predef.label_bg_separation, 0, 8, 1, 1, QtCore.Qt.AlignRight) # only visible in background-separated mode
        grid_bot.addWidget(Predef.checkbox_bg_separation, 0, 9, 1, 1, QtCore.Qt.AlignLeft) # only visible in background-separated mode
        grid_bot.addWidget(self.label_display, 0, 10, 1, 1, QtCore.Qt.AlignRight)
        grid_bot.addWidget(self.entry_logostyle, 0, 11, 1, 1, QtCore.Qt.AlignCenter)
        grid_bot.addWidget(self.entry_yscale, 0, 12, 1, 1, QtCore.Qt.AlignLeft)

        
        widget_top = QWidget()
        widget_top.setLayout(grid_top)
        widget_bot = QWidget()
        widget_bot.setLayout(grid_bot)
        layout.addWidget(widget_top)
        layout.addWidget(widget_bot)
        self.setLayout(layout)

        # Initialize with first cluster
        if Imports.embedding_fname != '' or Imports.linkage_fname != '':
            Predef.cluster_idx = 0
            Predef.k_idxs = np.array(Imports.clusters[Imports.cluster_col].loc[Imports.clusters[Imports.cluster_col] == 0].index)
            Predef.num_seqs = len(Predef.k_idxs)
            self.update_logo()  # Show initial logo for first cluster

    def onclick(self, event):
        try:
            predef_table.close()
        except:
            pass
        try:
            predef_stats_window.close()
        except:
            pass
        zooming_panning = (self.figure.canvas.cursor().shape() != 0) # arrow is 0 when not zooming (2) or panning (9)
        if not zooming_panning or self.clicked == False:
            if self.clicked == True:
                ix, iy = event.xdata, event.ydata if event is not None else (None, None)
                if ix != None and iy != None:
                    self.coordsX = float(ix)
                    self.coordsY = float(iy)
                
                # Handle both embedding and linkage cases
                if hasattr(self, 'embedding') and self.embedding is not None:
                    # Embedding case - use KDTree for nearest neighbor search
                    self.dist, idx = spatial.KDTree(self.embedding).query(np.array([self.coordsX, self.coordsY]))
                else:
                    # Linkage case - use simple distance-based selection
                    # Find the closest point in the scatter plot
                    coords = np.column_stack([self.pts_origX, self.pts_origY])
                    distances = np.sqrt((coords[:, 0] - self.coordsX)**2 + (coords[:, 1] - self.coordsY)**2)
                    idx = np.argmin(distances)
                    self.dist = distances[idx]
                    
            if self.dist < .01 or self.clicked == False:
                # Locate all members in dataframe that belong to cluster
                if self.clicked == True:
                    self.cluster_idx = Imports.clusters[Imports.cluster_col][idx]
                    self.k_idxs =  np.array(Imports.clusters[Imports.cluster_col].loc[Imports.clusters[Imports.cluster_col] == self.cluster_idx].index)
                    self.num_seqs = len(self.k_idxs)
                    self.embedding is not None
                    self.choose_cluster.blockSignals(True) # prevent feedback loop
                    self.choose_cluster.setValue(self.cluster_idx)
                    self.choose_cluster.blockSignals(False)
                # Redraw and resize figure
                self.ax.clear()
                
                embedding_selected = Imports.checkbox_embedding.isChecked() and Imports.embedding_fname != ''

                if embedding_selected:
                    # Use EmbeddingPlotter for embedding pathway
                    EmbeddingPlotter.plot_embedding_scatter(
                        ax=self.ax,
                        embedding=np.column_stack([self.pts_origX, self.pts_origY]),  # Use pts_origX and pts_origY like working version
                        cluster_assignments=self.cluster_colors,
                        cluster_idx=self.cluster_idx,
                        k_idxs=self.k_idxs,
                        plt_step=Custom.plt_step,
                        plt_marker=Custom.plt_marker,
                        plt_lw=Custom.plt_lw,
                        theme_color1='lightgray',
                        theme_color2='black'
                    )
                else:
                    # Use DendrogramPlotter for linkage pathway
                    DendrogramPlotter.plot_dendrogram(
                        ax=self.ax,
                        linkage=Imports.linkage,
                        maps=Imports.maps,
                        spin_cut_param=Imports.spin_cut_param,
                        combo_cut_criterion=Imports.combo_cut_criterion,
                    )
                    
                self.ax.tick_params(axis='both', which='major', labelsize=4)
                self.ax.get_xaxis().set_ticks([])
                self.ax.get_yaxis().set_ticks([])
                
                # Set appropriate title based on whether in embedding or linkage mode
                if hasattr(Imports, 'embedding_fname') and Imports.embedding_fname != '':
                    # Embedding mode - we loaded an actual embedding file
                    self.ax.set_title('Click on a cluster to view its logo', fontsize=5)
                else:
                    # Linkage mode - using placeholder coordinates
                    self.ax.set_title('Dendrogram-based clustering', fontsize=5)
                
                #self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % int(Imports.psi_1+1), fontsize=6) TODO - weird cropping
                #self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % int(Imports.psi_2+1), fontsize=6) TODO
                self.ax.autoscale()
                self.ax.margins(x=0)
                self.ax.margins(y=0)
                
                # Force the matplotlib toolbar to update its status
                if self.clicked == True:
                    self.canvas.motion_notify_event(*self.ax.transAxes.transform([0,0]))
                    self.canvas.motion_notify_event(*self.ax.transAxes.transform([ix,iy]))
                
                # Save sequences in cluster
                seqs = Imports.mave['Sequence'].str.slice(Imports.seq_start, Imports.seq_stop)
                self.seqs_cluster = seqs[self.k_idxs]
                seq_array_cluster = motifs.create(self.seqs_cluster, alphabet=Imports.alphabet)
                self.pfm_cluster = seq_array_cluster.counts # position frequency matrix
                    
                # Update sequence logo AFTER all other operations to prevent feedback loop
                self.update_logo()
                
                self.canvas.draw() # refresh canvas
                
                # Clear the processing flag
                if self.clicked == True:
                    self._processing_click = False

    def highlight_cluster(self):
        self.cluster_idx = int(self.choose_cluster.value())
        self.k_idxs =  np.array(Imports.clusters[Imports.cluster_col].loc[Imports.clusters[Imports.cluster_col] == self.cluster_idx].index)
        self.num_seqs = len(self.k_idxs)
        
        # For embedding mode, use the existing logic
        self.clicked = False  # bypass mouse click event to force replot
        self.dist = 9001
        self.onclick(None)
        self.clicked = True


    def open_inter_stats_window(self):
        global inter_stats_window
        try:
            inter_stats_window.close()
        except:
            pass
        inter_stats_window = InterStats()
        inter_stats_window.setMinimumSize(10, 10)
        inter_stats_window.show()


class AllStats(QtWidgets.QMainWindow):
    row = 0
    col = 0
    val = 0
    threshold = 100
    delta = False
    current_xlim = None  # Store current x-axis limits for zoom preservation
    current_ylim = None  # Store current y-axis limits for zoom preservation

    def __init__(self):
        super(AllStats, self).__init__()
        self.left = 10
        self.top = 10

        def open_marginals_window(self):
            global marginals_window
            try:
                marginals_window.close()
            except:
                pass
            marginals_window = Marginals()
            marginals_window.setMinimumSize(10, 10)
            marginals_window.show()

        AllStats.figure = Figure(dpi=dpi, constrained_layout=True)
        AllStats.figure.set_tight_layout(True)
        AllStats.canvas = FigureCanvas(AllStats.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        unwanted_buttons = ['Subplots']
        for x in self.toolbar.actions():
            if x.text() in unwanted_buttons:
                self.toolbar.removeAction(x)
        self.updateGeometry()
        centralwidget = QWidget()
        self.setCentralWidget(centralwidget)

        layout = QGridLayout(centralwidget)
        layout.setSizeConstraint(QGridLayout.SetMinimumSize) 
        layout.addWidget(self.toolbar, 0, 0, 1, 10)
        layout.addWidget(AllStats.canvas, 1 ,0, 50, 10)

        self.label_compare = QLabel('Metric: ')
        AllStats.combo_compare = QComboBox(self)
        AllStats.combo_compare.addItem('Positional Shannon entropy')
        AllStats.combo_compare.addItem('Percent mismatches to reference')
        AllStats.combo_compare.addItem('Consensus per cluster')
        AllStats.combo_compare.currentTextChanged.connect(self.reset)
        AllStats.combo_compare.setToolTip('Select the reference scheme for comparing the sequences in each cluster.')
        #if Imports.combo_ref.currentText() == 'None':
            #AllStats.combo_compare.setCurrentIndex(1)
            #AllStats.combo_compare.setItemData(0, False, QtCore.Qt.UserRole - 1)
        if Predef.df['Reference'].isnull().all():
            AllStats.combo_compare.setItemData(1, False, QtCore.Qt.UserRole - 1)

        self.btn_marginals = QPushButton('       Marginal distributions       ')
        self.btn_marginals.setDisabled(False)
        self.btn_marginals.setDefault(False)
        self.btn_marginals.setAutoDefault(False)
        self.btn_marginals.clicked.connect(open_marginals_window)
        self.btn_marginals.setToolTip('View marginal distributions.')
        
        layout.addWidget(self.label_compare, 51, 0, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(AllStats.combo_compare, 51, 1, 1, 1, QtCore.Qt.AlignLeft)
        layout.addWidget(self.btn_marginals, 51, 8, 1, 1, QtCore.Qt.AlignLeft)

        tabs.setTabEnabled(0, False) #freezes out 1st tab
        tabs.setTabEnabled(1, False) #freezes out 2nd tab
        tabs.setTabEnabled(2, False) #freezes out parent tab

        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        self.resize(int(self.width*(10/10.)), int(self.height*(8/10.)))
        self.setWindowTitle('Sensitivity of clusters to sequence elements')
        self.show()
      
        AllStats.ax, AllStats.cax, AllStats.reordered_ind, AllStats.revels = Imports.meta_explainer.plot_msm(
            column=Predef.df_col,
            delta_entropy=AllStats.delta,
            square_cells=False,
            gui=True,
            gui_figure=AllStats.figure
        )
        AllStats.canvas.mpl_connect('button_press_event', self.onclick)
        AllStats.ax.format_coord = self.format_coord
        AllStats.canvas.draw()

    def format_coord(self, x, y):
        col = int(np.floor(x))
        # Use sequential cluster number (0, 1, 2, ...) for display
        sequential_row = int(np.floor(y))
        val = AllStats.revels.iloc[int(np.floor(y))][col]
        
        # Calculate median activity for the cluster
        if hasattr(Imports, 'meta_explainer') and Imports.meta_explainer is not None:
            # Get the original cluster index if sorting is used
            if Imports.meta_explainer.cluster_order is not None:
                original_cluster_idx = Imports.meta_explainer.cluster_order[int(np.floor(y))]
            else:
                original_cluster_idx = int(np.floor(y))
            
            # Get sequences in this cluster and calculate median DNN score
            cluster_seqs = Imports.meta_explainer.mave[Imports.meta_explainer.mave['Cluster'] == original_cluster_idx]
            median_activity = cluster_seqs['DNN'].median()
            median_str = f', median activity={median_activity:.2f}'
        else:
            median_str = ''
        
        if Predef.df_col == 'Reference' or Predef.df_col == 'Consensus':
            return f'cluster={sequential_row:1.0f}, position={col:1.0f}, value={abs(val):1.1f}%{median_str}'
        elif Predef.df_col == 'Entropy':
            return f'cluster={sequential_row:1.0f}, position={col:1.0f}, value={abs(val):1.2f} bits{median_str}'
        
    def reset(self):
        # Store current view limits before clearing (only if this is not the initial creation)
        if hasattr(AllStats, 'ax') and AllStats.ax is not None:
            AllStats.current_xlim = AllStats.ax.get_xlim()
            AllStats.current_ylim = AllStats.ax.get_ylim()
        
        try:
            marginals_window.close()
        except:
            pass
        if AllStats.combo_compare.currentText() == 'Percent mismatches to reference':
            Predef.df_col = 'Reference'
            AllStats.delta = False
        elif AllStats.combo_compare.currentText() == 'Consensus per cluster':
            Predef.df_col = 'Consensus'
            AllStats.delta = False
        elif AllStats.combo_compare.currentText() == 'Positional Shannon entropy':
            Predef.df_col = 'Entropy'
            AllStats.delta = False

        AllStats.ax.clear()
        AllStats.cax.cla()
        AllStats.figure.clear()
        AllStats.ax, AllStats.cax, AllStats.reordered_ind, AllStats.revels = Imports.meta_explainer.plot_msm(
            column=Predef.df_col,
            delta_entropy=AllStats.delta,
            square_cells=False,
            gui=True,
            gui_figure=AllStats.figure
        )
        AllStats.ax.format_coord = self.format_coord
        
        # Restore view limits if they exist (preserves zoom state when switching metrics)
        if AllStats.current_xlim is not None and AllStats.current_ylim is not None:
            AllStats.ax.set_xlim(AllStats.current_xlim)
            AllStats.ax.set_ylim(AllStats.current_ylim)
        
        AllStats.canvas.draw()

    def open_cell_window(self):
        global allstats_cell_window
        try:
            allstats_cell_window.close()
        except:
            pass
        allstats_cell_window = Cell()
        allstats_cell_window.setMinimumSize(10, 10)
        allstats_cell_window.show()

    def onclick(self, event):
        zooming_panning = (self.figure.canvas.cursor().shape() != 0) # arrow is 0 when not zooming (2) or panning (9)
        if not zooming_panning:
            ix, iy = event.xdata, event.ydata
            if ix != None and iy != None:
                AllStats.col = int(np.floor(float(ix)))
                AllStats.row = AllStats.reordered_ind[int(np.floor(float(iy)))]
                val = AllStats.revels.iloc[int(np.floor(iy))][AllStats.col]
                if AllStats.combo_compare.currentText() == 'Percent mismatches to reference':
                    AllStats.val = f'{abs(val):1.1f}%'
                elif AllStats.combo_compare.currentText() == 'Positional Shannon entropy':
                    AllStats.val = f'{abs(val):1.1f} bits'
                else:  # Consensus per cluster
                    AllStats.val = f'{abs(val):1.1f}'
                self.open_cell_window()

    def closeEvent(self, ce): # activated when user clicks to exit via subwindow button
        # Reset zoom state when closing window
        AllStats.current_xlim = None
        AllStats.current_ylim = None
        self.reset()
        try:
            allstats_cell_window.close()
        except:
            pass
        try:
            marginals_window.close()
        except:
            pass
        tabs.setTabEnabled(0, True)
        if Imports.embedding_fname != '':
            tabs.setTabEnabled(1, True)
        else:
            tabs.setTabEnabled(1, False)
        tabs.setTabEnabled(2, True)
        Predef.btn_all_stats.setDisabled(False)


class InterStats(QtWidgets.QMainWindow):
    def __init__(self):
        super(InterStats, self).__init__()
        self.left = 10
        self.top = 10

        InterStats.figure = Figure(dpi=dpi, constrained_layout=True)
        InterStats.figure.set_tight_layout(True)
        InterStats.canvas = FigureCanvas(InterStats.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        unwanted_buttons = ['Subplots']
        for x in self.toolbar.actions():
            if x.text() in unwanted_buttons:
                self.toolbar.removeAction(x)
        self.updateGeometry()
        centralwidget = QWidget()
        self.setCentralWidget(centralwidget)

        layout = QGridLayout(centralwidget)
        layout.setSizeConstraint(QGridLayout.SetMinimumSize) 
        layout.addWidget(self.toolbar, 0, 0, 1, 10)
        layout.addWidget(InterStats.canvas, 1, 0, 50, 10)

        # Add controls for plot options
        self.label_plot_type = QLabel('Plot type: ')
        InterStats.combo_plot_type = QComboBox(self)
        InterStats.combo_plot_type.addItem('Box plot')
        InterStats.combo_plot_type.addItem('Bar plot')
        
        # Function to handle plot type changes and update metric state
        def on_plot_type_changed():
            current_metric = InterStats.combo_metric.currentText()
            if InterStats.combo_plot_type.currentText() == 'Box plot':
                # For box plots, remove occupancy option if it exists
                if InterStats.combo_metric.count() > 1:
                    InterStats.combo_metric.removeItem(1)
                # If occupancy was selected, switch to prediction
                if current_metric == 'Occupancy':
                    InterStats.combo_metric.setCurrentIndex(0)
            else:
                # For bar plots, add occupancy option if it doesn't exist
                if InterStats.combo_metric.count() == 1:
                    InterStats.combo_metric.addItem('Occupancy')
                # Restore occupancy selection if it was previously selected
                if current_metric == 'Occupancy':
                    InterStats.combo_metric.setCurrentIndex(1)            
            self.reset()
        
        InterStats.combo_plot_type.currentTextChanged.connect(on_plot_type_changed)
        InterStats.combo_plot_type.setToolTip('Select the type of visualization for cluster statistics.')

        self.label_metric = QLabel('Metric: ')
        InterStats.combo_metric = QComboBox(self)
        InterStats.combo_metric.addItem('Prediction')
        # Don't add Occupancy initially since Box plot is default
        InterStats.combo_metric.currentTextChanged.connect(self.reset)
        InterStats.combo_metric.setToolTip('Select what to visualize (DNN predictions or cluster occupancy).')

        self.label_show_ref = QLabel('Show reference: ')
        InterStats.checkbox_show_ref = QCheckBox(self)
        InterStats.checkbox_show_ref.setChecked(True)
        InterStats.checkbox_show_ref.stateChanged.connect(self.reset)
        InterStats.checkbox_show_ref.setToolTip('Highlight the reference sequence cluster if available.')

        self.label_show_fliers = QLabel('Show outliers: ')
        InterStats.checkbox_show_fliers = QCheckBox(self)
        InterStats.checkbox_show_fliers.setChecked(False)
        InterStats.checkbox_show_fliers.setEnabled(False)  # Disabled by default since compact is checked
        InterStats.checkbox_show_fliers.stateChanged.connect(self.reset)
        InterStats.checkbox_show_fliers.setToolTip('Show outlier points in box plots.')

        self.label_compact = QLabel('Compact visualization: ')
        InterStats.checkbox_compact = QCheckBox(self)
        InterStats.checkbox_compact.setChecked(True)
        InterStats.checkbox_compact.stateChanged.connect(self.reset)
        InterStats.checkbox_compact.setToolTip('Use compact representation for box plots (dots and IQR lines instead of full boxplots).')
        
        # Function to handle compact visualization changes and update outlier state
        def on_compact_changed():
            if InterStats.checkbox_compact.isChecked():
                # Disable and uncheck outliers when compact is enabled
                InterStats.checkbox_show_fliers.setChecked(False)
                InterStats.checkbox_show_fliers.setEnabled(False)
            else:
                # Enable outliers when compact is disabled (but keep unchecked)
                InterStats.checkbox_show_fliers.setEnabled(True)
            # Then reset the plot
            self.reset()
        
        InterStats.checkbox_compact.stateChanged.disconnect()
        InterStats.checkbox_compact.stateChanged.connect(on_compact_changed)
        
        layout.addWidget(self.label_plot_type, 51, 0, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(InterStats.combo_plot_type, 51, 1, 1, 1, QtCore.Qt.AlignLeft)
        layout.addWidget(self.label_metric, 51, 2, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(InterStats.combo_metric, 51, 3, 1, 1, QtCore.Qt.AlignLeft)
        layout.addWidget(self.label_show_ref, 51, 4, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(InterStats.checkbox_show_ref, 51, 5, 1, 1, QtCore.Qt.AlignLeft)
        layout.addWidget(self.label_show_fliers, 51, 6, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(InterStats.checkbox_show_fliers, 51, 7, 1, 1, QtCore.Qt.AlignLeft)
        layout.addWidget(self.label_compact, 51, 8, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(InterStats.checkbox_compact, 51, 9, 1, 1, QtCore.Qt.AlignLeft)

        tabs.setTabEnabled(0, False) #freezes out 1st tab
        tabs.setTabEnabled(1, False) #freezes out 2nd tab
        tabs.setTabEnabled(2, False) #freezes out parent tab

        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        self.resize(int(self.width*(8/10.)), int(self.height*(6/10.)))
        self.setWindowTitle('Cluster Activity Distributions')
        self.show()
        
        # Initialize metric state based on initial plot type (Box plot is default)
        InterStats.combo_metric.setItemData(1, False, QtCore.Qt.UserRole - 1)
        
        # Generate the initial plot
        self.reset()

    def reset(self):
        """Reset and regenerate the cluster activities plot."""
        # Check if meta_explainer is available
        if hasattr(Imports, 'meta_explainer') and Imports.meta_explainer is not None:
            # Clear the existing figure
            InterStats.figure.clear()
            ax = InterStats.figure.add_subplot(111)
            
            # Get plot parameters from GUI controls
            plot_type = 'box' if InterStats.combo_plot_type.currentText() == 'Box plot' else 'bar'
            metric = 'prediction' if InterStats.combo_metric.currentText() == 'Prediction' else 'occupancy'
            show_ref = InterStats.checkbox_show_ref.isChecked()
            show_fliers = InterStats.checkbox_show_fliers.isChecked()
            compact = InterStats.checkbox_compact.isChecked()
            
            # Collect data for each cluster (similar to plot_cluster_stats but without creating new figure)
            boxplot_data = []
            
            # Use actual clusters from data instead of cluster_indices
            actual_clusters = np.sort(Imports.meta_explainer.mave['Cluster'].unique())
            cluster_to_idx = {k: i for i, k in enumerate(actual_clusters)}
            
            for k in actual_clusters:
                k_idxs = Imports.meta_explainer.mave.loc[Imports.meta_explainer.mave['Cluster'] == k].index
                if plot_type == 'box' or metric == 'prediction':
                    data = Imports.meta_explainer.mave.loc[k_idxs, 'DNN']
                    boxplot_data.append(data)
                else:  # counts for bar plot
                    boxplot_data.append([len(k_idxs)])
                    
            # Sort using class-level ordering if it exists
            if Imports.meta_explainer.cluster_order is not None:
                sorted_data = []
                for k in Imports.meta_explainer.cluster_order:
                    idx = cluster_to_idx[k]
                    sorted_data.append(boxplot_data[idx])
                boxplot_data = sorted_data
                
                # Update membership tracking
                mapping_dict = {old_k: new_k for new_k, old_k in 
                            enumerate(Imports.meta_explainer.cluster_order)}

            if plot_type == 'box':
                # Calculate IQR
                iqr_values = [np.percentile(data, 75) - np.percentile(data, 25) 
                            for data in boxplot_data if len(data) > 0]
                average_iqr = np.mean(iqr_values) if iqr_values else 0
                
                if not compact:
                    # Create boxplot using the existing axis
                    ax.boxplot(boxplot_data[::-1], vert=False, 
                            showfliers=show_fliers, 
                            medianprops={'color': 'black'},
                            flierprops={'marker': 'o', 'markersize': 0.5, 'markerfacecolor': 'black', 'markeredgecolor': 'black'})
                    ax.set_yticks(range(1, len(boxplot_data) + 1)[::10])
                    ax.set_yticklabels(range(len(boxplot_data))[::-1][::10], fontsize=4)
                else:
                    # Compact representation
                    for pos, values in enumerate(boxplot_data[::-1]):
                        values = np.array(values)            
                        median = np.median(values)
                        q1 = np.percentile(values, 25)
                        q3 = np.percentile(values, 75)
                        ax.plot([q1, q3], [pos+1, pos+1], color='gray', lw=.5)  # plot the IQR line
                        ax.plot(median, pos+1, 'o', color='k', markersize=1, zorder=100)  # plot the median point
                    ax.set_yticks(range(1, len(boxplot_data) + 1)[::10])
                    ax.set_yticklabels(range(len(boxplot_data))[::-1][::10], fontsize=4)
                
                ax.set_ylabel('Clusters', fontsize=6)
                ax.set_xlabel('Activity', fontsize=6)
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax.set_title(f'Average IQR: {average_iqr:.2f}', fontsize=6)
                
                # Update reference cluster index if sorting is enabled
                if show_ref and Imports.meta_explainer.ref_seq is not None:
                    ref_cluster = Imports.meta_explainer.membership_df.loc[Imports.meta_explainer.ref_idx, 'Cluster']
                    if Imports.meta_explainer.cluster_order is not None:
                        ref_cluster = mapping_dict[ref_cluster]
                    ref_data = boxplot_data[ref_cluster]
                    if len(ref_data) > 0:
                        ax.axvline(np.median(ref_data), c='red', 
                                label='Ref', zorder=-100)
                        ax.legend(loc='best', fontsize=4, frameon=False)

            else:  # bar plot
                y_positions = np.arange(len(boxplot_data))
                values = [np.median(data) if metric == 'prediction' else data[0] 
                        for data in boxplot_data]
                height = 1.0
                
                if show_ref and Imports.meta_explainer.ref_seq is not None:
                    ref_cluster = Imports.meta_explainer.membership_df.loc[Imports.meta_explainer.ref_idx, 'Cluster']
                    if Imports.meta_explainer.cluster_order is not None:
                        ref_cluster = mapping_dict[ref_cluster]
                    colors = ['red' if i == ref_cluster else 'C0' 
                            for i in range(len(values))]
                    ax.barh(y_positions, values, height=height, color=colors)
                else:
                    ax.barh(y_positions, values, height=height)
                
                ax.set_yticks(y_positions[::10])
                ax.set_yticklabels(y_positions[::10], fontsize=4)
                ax.set_ylabel('Cluster', fontsize=6)
                ax.set_xlabel('Activity' if metric == 'prediction' else 'Count', fontsize=6)
                ax.invert_yaxis()
                ax.axvline(x=0, color='black', linewidth=0.5, zorder=100)
            
            # Set consistent tick parameters to match other plots
            ax.tick_params(axis="x", labelsize=4)
            ax.tick_params(axis="y", labelsize=4)
            
            InterStats.figure.tight_layout()
            InterStats.canvas.draw()
            
        else:
            # Fallback if no meta_explainer available
            InterStats.figure.clear()
            ax = InterStats.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No cluster data available.\nPlease load and process data first.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            InterStats.canvas.draw()

    def closeEvent(self, ce):
        """Handle window close event."""
        tabs.setTabEnabled(0, True)
        if Imports.embedding_fname != '':
            tabs.setTabEnabled(1, True)
        else:
            tabs.setTabEnabled(1, False)
        tabs.setTabEnabled(2, True)


class Cell(QtWidgets.QMainWindow):
    def __init__(self):
        super(Cell, self).__init__()
        self.left = 10
        self.top = 10
        self.figure = Figure(dpi=dpi, constrained_layout=True)
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        unwanted_buttons = ['Subplots']
        for x in self.toolbar.actions():
            if x.text() in unwanted_buttons:
                self.toolbar.removeAction(x)
        self.updateGeometry()
        centralwidget = QWidget()
        self.setCentralWidget(centralwidget)
        layout = QGridLayout(centralwidget)
        layout.setSizeConstraint(QGridLayout.SetMinimumSize)
        layout.addWidget(self.toolbar, 0, 0, 1, 12)
        layout.addWidget(self.canvas, 1 ,0, 50, 12)
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        
        # Calculate the remapped cluster number for the window title
        if Imports.cluster_reverse_mapping is not None:
            title_cluster_num = Imports.cluster_reverse_mapping[AllStats.row]
        else:
            title_cluster_num = AllStats.row
            
        if AllStats.combo_compare.currentText() == 'Percent mismatches to reference':
            self.setWindowTitle('Cluster: %s | Position: %s | Mismatch: %s' % (title_cluster_num, AllStats.col, AllStats.val))
        elif AllStats.combo_compare.currentText() == 'Positional Shannon entropy':
            self.setWindowTitle('Cluster: %s | Position: %s | Entropy: %s' % (title_cluster_num, AllStats.col, AllStats.val))
        else:  # Consensus per cluster
            self.setWindowTitle('Cluster: %s | Position: %s | Match: %s' % (title_cluster_num, AllStats.col, AllStats.val))
        self.resize(int(self.width*(5/10.)), int(self.height*(5/10.)))
        self.show()

        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('Character', fontsize=4)
        self.ax.set_ylabel('Counts', fontsize=4)
        self.ax.tick_params(axis="x", labelsize=4)
        self.ax.tick_params(axis="y", labelsize=4)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        for k in Predef.clusters_idx:
            # Use reverse mapping to convert original cluster index to remapped index
            if Imports.cluster_reverse_mapping is not None:
                remapped_cluster_idx = Imports.cluster_reverse_mapping[AllStats.row]
            else:
                remapped_cluster_idx = AllStats.row
            k_idxs = Imports.mave.loc[Imports.mave['Cluster'] == remapped_cluster_idx].index

        seqs = Imports.mave['Sequence'].str.slice(Imports.seq_start, Imports.seq_stop)
        seqs_cluster = seqs[k_idxs]
        occ = seqs_cluster.str.slice(AllStats.col, AllStats.col+1)
        self.vc = occ.value_counts()
        self.vc = self.vc.sort_index()
        self.vc.plot(kind='bar', ax=self.ax, rot=0)
        if AllStats.combo_compare.currentText() == 'Percent mismatches to reference':
            ref_short = Imports.ref_full[Imports.seq_start:Imports.seq_stop]
            self.ax.set_title('Reference: %s' % ref_short[AllStats.col:AllStats.col+1], fontsize=4)
        else:
            seq_array_cluster = motifs.create(seqs_cluster, alphabet=Imports.alphabet)
            pfm_cluster = seq_array_cluster.counts # position frequency matrix
            consensus_seq = pfm_cluster.consensus
            self.ax.set_title('Consensus: %s' % consensus_seq[AllStats.col:AllStats.col+1], fontsize=4)
        self.ax.format_coord = self.format_coord
        self.canvas.draw()

    def format_coord(self, x, y):
        x = round(x)
        y = round(self.vc[x])
        return f'counts={y:1.0f}'


class Marginals(QtWidgets.QMainWindow):
    def __init__(self):
        super(Marginals, self).__init__()
        self.left = 10
        self.top = 10

        def choose_threshold():
            AllStats.threshold = self.spin_threshold.value()
            self.reset()

        self.figure = Figure(dpi=dpi, constrained_layout=True)
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        unwanted_buttons = ['Subplots']
        for x in self.toolbar.actions():
            if x.text() in unwanted_buttons:
                self.toolbar.removeAction(x)
        self.updateGeometry()
        centralwidget = QWidget()
        self.setCentralWidget(centralwidget)

        layout = QGridLayout(centralwidget)
        layout.setSizeConstraint(QGridLayout.SetMinimumSize) 
        layout.addWidget(self.toolbar, 0, 0, 1, 19)
        layout.addWidget(self.canvas, 1 ,0, 50, 19)

        self.label_threshold = QLabel('Choose threshold: ')
        self.combo_inequality = QComboBox(self)
        self.combo_inequality.addItem('%s' % u"\u2265") # unicode: greater than or equal
        self.combo_inequality.addItem('%s' % u"\u2264") # unicode: less than or equal
        #self.combo_inequality.addItem('%s' % u"\u003e") # unicode: greater than
        #self.combo_inequality.addItem('%s' % u"\u003c") # unicode: less than
        #self.combo_inequality.addItem('%s' % u"\u003d") # unicode: equals
        self.combo_inequality.currentTextChanged.connect(choose_threshold)
        self.combo_inequality.setToolTip('Select the threshold inequality.')

        self.spin_threshold = QDoubleSpinBox(self)
        self.spin_threshold.setMinimum(0)
        if Predef.df_col == 'Reference' or Predef.df_col == 'Consensus':
            self.spin_threshold.setMaximum(100)
            self.spin_threshold.setSuffix('%')
            AllStats.threshold = 50
            self.spin_threshold.setDecimals(0)
        elif Predef.df_col == 'Entropy':
            self.spin_threshold.setMaximum(2)
            self.spin_threshold.setSuffix(' bits')
            AllStats.threshold = 1
            self.spin_threshold.setDecimals(2)
        self.spin_threshold.setValue(AllStats.threshold)
        self.spin_threshold.lineEdit().returnPressed.connect(choose_threshold) # only signal when the Return or Enter key is pressed
        self.spin_threshold.setToolTip('Set the threshold for counting instances in the rows/columns.')

        self.btn_threshold = QPushButton('View')
        self.btn_threshold.setDisabled(False)
        self.btn_threshold.setDefault(False)
        self.btn_threshold.setAutoDefault(False)
        self.btn_threshold.clicked.connect(choose_threshold)
        self.btn_threshold.setToolTip('Update the plot above based on the threshold inputs.')

        layout.addWidget(QVLine(), 51, 0, 1, 8, QtCore.Qt.AlignCenter)
        layout.addWidget(self.label_threshold, 51, 8, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.combo_inequality, 51, 9, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.spin_threshold, 51, 10, 1, 1, QtCore.Qt.AlignCenter)
        layout.addWidget(self.btn_threshold, 51, 11, 1, 1, QtCore.Qt.AlignLeft)
        layout.addWidget(QVLine(), 51, 12, 1, 8, QtCore.Qt.AlignCenter)

        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        self.setWindowTitle('Marginal distributions | %s' % AllStats.combo_compare.currentText())
        self.resize(int(self.width*(9./10.)), int(self.height*(5/10.)))
        self.show()

        df = AllStats.revels.copy()
        self.nP = Predef.df['Position'].max()+1
        self.nC = Predef.df['Cluster'].max()+1
        df = (df >= AllStats.threshold).astype(int) # binary mask
        #df = df.mask(df < 100) # non-binary mask
        self.occ_position = df.sum(axis=0).to_numpy()
        self.occ_cluster = df.sum(axis=1).to_numpy()

        self.ax1 = self.figure.add_subplot(211)
        self.ax1.bar(range(len(self.occ_position)), self.occ_position)
        self.ax1.spines[['right', 'top']].set_visible(False)
        self.ax1.set_xlabel('Position', fontsize=4)
        self.ax1.set_ylabel('Counts', fontsize=4)
        self.ax1.tick_params(axis="x", labelsize=4)
        self.ax1.tick_params(axis="y", labelsize=4)
        self.ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax1.set_xlim(-.5, self.nP-.5)
        self.ax1.format_coord = self.format_coord1

        self.ax2 = self.figure.add_subplot(212)
        self.ax2.bar(range(len(self.occ_cluster)), self.occ_cluster)
        self.ax2.spines[['right', 'top']].set_visible(False)
        self.ax2.set_xlabel('Cluster', fontsize=4)
        self.ax2.set_ylabel('Counts', fontsize=4)
        self.ax2.tick_params(axis="x", labelsize=4)
        self.ax2.tick_params(axis="y", labelsize=4)
        self.ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax2.set_xlim(-.5, self.nC-.5)
        self.ax2.format_coord = self.format_coord2
        self.ctick_labels = []

        for c_idx in range(self.nC):
            self.ctick_labels.append(str(int(AllStats.reordered_ind[c_idx])))
        ctick_labels_sparse = []
        if self.nC > 10:
            c_skip = 10
        else:
            c_skip = 1
        ctick_range = np.arange(0, self.nC, c_skip)
        c_idx = 0
        for i in ctick_range:
            if int(i)%c_skip == 0:
                ctick_labels_sparse.append(str(int(AllStats.reordered_ind[c_idx])))
            c_idx += c_skip
        self.ax2.set_xticks(ctick_range)
        self.ax2.set_xticklabels(ctick_labels_sparse, rotation=0, minor=False)

        self.canvas.draw()


    def format_coord1(self, x, y):
        x = round(x)
        y = round(self.occ_position[x])
        return f'position={x:1.0f}, counts={y:1.0f}'
    
    def format_coord2(self, x, y):
        x = round(x)
        y = round(self.occ_cluster[x])
        x = round(int(self.ctick_labels[x]))
        return f'cluster={x:1.0f}, counts={y:1.0f}'
    
    def reset(self):
        self.ax1.clear()
        self.ax2.clear()
        self.figure.clear()

        df = AllStats.revels.copy()
        self.nP = Predef.df['Position'].max()+1
        self.nC = Predef.df['Cluster'].max()+1

        # Apply binary mask (counts)
        if self.combo_inequality.currentText() == ('%s' % u"\u2265"):
            df = (df >= AllStats.threshold).astype(int)
        elif self.combo_inequality.currentText() == ('%s' % u"\u2264"):
            df = (df <= AllStats.threshold).astype(int)
        '''elif self.combo_inequality.currentText() == ('%s' % u"\u003e"):
            df = (df > AllStats.threshold).astype(int)
        elif self.combo_inequality.currentText() == ('%s' % u"\u003c"):
            df = (df < AllStats.threshold).astype(int)
        elif self.combo_inequality.currentText() == ('%s' % u"\u003d"):
            df = (df == AllStats.threshold).astype(int)'''

        self.occ_position = df.sum(axis=0).to_numpy()
        self.occ_cluster = df.sum(axis=1).to_numpy()

        self.ax1 = self.figure.add_subplot(211)
        self.ax1.bar(range(len(self.occ_position)), self.occ_position)
        self.ax1.spines[['right', 'top']].set_visible(False)
        self.ax1.set_xlabel('Position', fontsize=4)
        self.ax1.set_ylabel('Counts', fontsize=4)
        self.ax1.tick_params(axis="x", labelsize=4)
        self.ax1.tick_params(axis="y", labelsize=4)
        self.ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax1.set_xlim(-.5, self.nP-.5)
        self.ax1.format_coord = self.format_coord1

        self.ax2 = self.figure.add_subplot(212)
        self.ax2.bar(range(len(self.occ_cluster)), self.occ_cluster)
        self.ax2.spines[['right', 'top']].set_visible(False)
        self.ax2.set_xlabel('Cluster', fontsize=4)
        self.ax2.set_ylabel('Counts', fontsize=4)
        self.ax2.tick_params(axis="x", labelsize=4)
        self.ax2.tick_params(axis="y", labelsize=4)
        self.ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax2.set_xlim(-.5, self.nC-.5)
        self.ax2.format_coord = self.format_coord2
        self.ctick_labels = []
        for c_idx in range(self.nC):
            self.ctick_labels.append(str(int(AllStats.reordered_ind[c_idx])))
        ctick_labels_sparse = []
        if self.nC > 10:
            c_skip = 10
        else:
            c_skip = 1
        ctick_range = np.arange(0, self.nC, c_skip)
        c_idx = 0
        for i in ctick_range:
            if int(i)%c_skip == 0:
                ctick_labels_sparse.append(str(int(AllStats.reordered_ind[c_idx])))
            c_idx += c_skip
        self.ax2.set_xticks(ctick_range)
        self.ax2.set_xticklabels(ctick_labels_sparse, rotation=0, minor=False)

        self.canvas.draw()
    

################################################################################
# Main window

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('%s' % progname)
        self.setWindowIcon(QtGui.QIcon(os.path.join(icon_dir, '256x256.png')))
        self.setGeometry(100, 100, 1200, 800)
        self.setFont(font_standard)

        # Create menu bar
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        
        # File menu
        fileMenu = mainMenu.addMenu('&File')
        aboutAction = QtWidgets.QAction('&About', self)
        aboutAction.triggered.connect(self.about)
        fileMenu.addAction(aboutAction)
        fileMenu.addSeparator()
        restartAction = QtWidgets.QAction('&Restart', self)
        restartAction.triggered.connect(self.fileRestart)
        fileMenu.addAction(restartAction)
        exitAction = QtWidgets.QAction('&Exit', self)
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        # Help menu
        helpMenu = mainMenu.addMenu('&Help')
        importHelpAction = QtWidgets.QAction('Import Files Help', self)
        importHelpAction.triggered.connect(self.show_import_help)
        helpMenu.addAction(importHelpAction)

        customHelpAction = QtWidgets.QAction('Custom Clusters Help', self)
        customHelpAction.triggered.connect(self.show_custom_help)
        helpMenu.addAction(customHelpAction)

        predefinedHelpAction = QtWidgets.QAction('Predefined Clusters Help', self)
        predefinedHelpAction.triggered.connect(self.show_predefined_help)
        helpMenu.addAction(predefinedHelpAction)

        style = """QTabWidget::tab-bar{
            alignment: center}"""

        tab0 = Imports(self)
        tab1 = Custom(self)
        tab2 = Predef(self)

        tab0.btn_process_imports.clicked.connect(self.process_imports)

        global tabs
        tabs = QtWidgets.QTabWidget(self)
        tabs.addTab(tab0, '       Import Files       ')
        tabs.addTab(tab1, '      Custom Clusters      ')
        tabs.addTab(tab2, '    Predefined Clusters    ')
        tabs.setTabEnabled(1, False)
        tabs.setTabEnabled(2, False)
        self.setStyleSheet(style)
        self.setCentralWidget(tabs)
        #self.showMaximized()

        tabs.currentChanged.connect(self.onTabChange) #signal for tab changed via direct click

        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        #self.resize(int(self.width*(1/2.)), self.height)
        self.resize(int(self.width*(9/10.)), int(self.height))
        self.show()

    def onTabChange(self, i):
        # Check for imports changes first
        if Imports.imports_changed and not Imports.imports_warning_acknowledged and i in [1, 2]:  # Custom or Predefined clusters tabs
            box = QMessageBox(self)
            box.setWindowTitle('%s Warning' % progname)
            box.setText('<b>Imports Not Updated</b>')
            box.setFont(font_standard)
            box.setIcon(QMessageBox.Warning)
            box.setInformativeText('Import settings have been modified but not yet applied.\n\nChanges will not be reflected until you return to the Import Files tab and click "Update Imports".')
            box.setStandardButtons(QMessageBox.Ok)
            box.exec_()
            Imports.imports_warning_acknowledged = True
        
        # Original onTabChange logic
        Imports.current_tab = i
        try:
            predef_table.close()
        except:
            pass
        try:
            predef_stats_window.close()
        except:
            pass

    def process_imports(self):
        # Clear P3 logo figure at the start of any import update
        try:
            tab2.logo_ax.clear()
            tab2.logo_ax.axis('off')
            tab2.logo_canvas.draw()
        except:
            pass

        try:
            # Clear embedding-specific state
            if hasattr(tab2, 'embedding'):
                delattr(tab2, 'embedding')
            if hasattr(tab2, 'seqs_cluster'):
                delattr(tab2, 'seqs_cluster')
            if hasattr(tab2, 'pfm_cluster'):
                delattr(tab2, 'pfm_cluster')
        except:
            pass

        # Reset all logo display checkboxes to unchecked state when reprocessing imports
        # Temporarily disconnect event handlers to prevent unwanted logo updates
        Predef.checkbox_variability.stateChanged.disconnect()
        Predef.checkbox_bg_separation.stateChanged.disconnect()
        Predef.checkbox_avg_background.stateChanged.disconnect()
        
        # Now safely uncheck the checkboxes
        Predef.checkbox_variability.setChecked(False)
        Predef.show_variability_logo = False
        Predef.show_background_separated = False
        Predef.show_average_background = False
        
        # Reconnect the event handlers using stored function references
        Predef.checkbox_variability.stateChanged.connect(Predef.toggle_variability_func)
        Predef.checkbox_bg_separation.stateChanged.connect(Predef.toggle_background_separation_func)
        Predef.checkbox_avg_background.stateChanged.connect(Predef.toggle_avg_background_func)
        
        # Check that either embedding OR linkage is selected (but not both)
        embedding_selected = Imports.checkbox_embedding.isChecked() and Imports.embedding_fname != ''
        linkage_selected = Imports.checkbox_linkage.isChecked() and Imports.linkage_fname != ''
        
        if (Imports.mave_fname != '' and Imports.maps_fname != '' and 
            (embedding_selected or linkage_selected) and 
            not (embedding_selected and linkage_selected)):
            
            Imports.imports_changed = False
            Imports.imports_confirmed = True
            Imports.imports_warning_acknowledged = False
            Imports.btn_process_imports.setText('Update Imports')
            Imports.btn_process_imports.setToolTip('Apply changes to imports before proceeding to analysis.')
            
            # Load in silico mave and (optionally) reference seqence
            Imports.mave = pd.read_csv(Imports.mave_fname)
            Imports.mave_col_names = list(Imports.mave)
            if 'GIA' not in Imports.mave_col_names:
                Custom.entry_cmap.setItemData(1, False, QtCore.Qt.UserRole - 1)
            if 'Hamming' not in Imports.mave_col_names:
                Custom.entry_cmap.setItemData(2, False, QtCore.Qt.UserRole - 1)
            if 'Task' not in Imports.mave_col_names:
                Custom.entry_cmap.setItemData(3, False, QtCore.Qt.UserRole - 1)
            Imports.nS = len(Imports.mave)
            print('Determining alphabet...')
            if 1: # search full dataset
                Imports.alphabet = sorted(set(''.join(Imports.mave['Sequence'])))
            else: # search subset of dataset
                Imports.alphabet = sorted(set(''.join(Imports.mave['Sequence'].iloc[0:100])))
            Imports.seq_start = 0
            Imports.seq_stop = len(Imports.mave['Sequence'][0])
            Imports.seq_length = Imports.seq_stop - Imports.seq_start
            # Remove redundant maps loading since it's already loaded in load_maps method
            # Imports.maps = np.load(Imports.maps_fname, allow_pickle=True)
            # Imports.maps_shape = Imports.maps.shape
            Imports.dim = (Imports.seq_length)*Imports.maps_shape[2]
            if 'Hamming' in Imports.mave_col_names:
                Imports.hamming_orig = Imports.mave['Hamming']

            Cluster.checkbox_ref = QCheckBox('Color reference', self)
            Cluster.checkbox_contrib = QCheckBox('Reference contribution', self)
            if Imports.combo_ref.currentText() == 'Custom' and (len(Imports.entry_ref.text()) != Imports.seq_length or all(c in Imports.alphabet for c in Imports.entry_ref.text()) is False):
                box = QMessageBox(self)
                box.setWindowTitle('%s Error' % progname)
                box.setText('<b>Input Error</b>')
                box.setFont(font_standard)
                box.setIcon(QMessageBox.Information)
                box.setInformativeText('Reference sequence has an incorrect length and/or contains an incorrect alphabet with respect to the MAVE dataset.')
                box.setStandardButtons(QMessageBox.Ok)
                box.setDefaultButton(QMessageBox.Ok)
                ret = box.exec_()
                Imports.process_imports.setDisabled(False)
                return

            Imports.ref_full = Imports.entry_ref.text()
            Cluster.checkbox_ref.setDisabled(False)
            if Imports.combo_ref.currentText() == 'None':
                Custom.logo_ref = False
                Cluster.checkbox_ref.setChecked(False)
                Cluster.checkbox_ref.setDisabled(True)
            else:
                Custom.logo_ref = True
                Cluster.checkbox_ref.setChecked(True)
            Custom.checkbox_ref.setDisabled(False)

            # Reset all widgets on page 2 (if altered before change to Imports page)
            Custom.entry_cmap.setCurrentIndex(0)
            Custom.entry_theme.setCurrentIndex(0)
            Custom.entry_zorder.setCurrentIndex(0)
            Custom.checkbox_ref.setChecked(False)
            Custom.entry_stepsize.setValue(0)
            Custom.entry_markersize.setValue(.5)
            Custom.entry_eig1.setValue(1)
            Custom.entry_eig2.setValue(2)

            if int(Imports.startbox_cropped.value()) != 0 and int(Imports.stopbox_cropped.value()) != len(Imports.mave['Sequence'][0]):
                Imports.map_crop = True
                Imports.seq_start = int(Imports.startbox_cropped.value())
                Imports.seq_stop = int(Imports.stopbox_cropped.value())
                Imports.seq_length = Imports.seq_stop - Imports.seq_start
                Imports.dim = (Imports.seq_length)*Imports.maps.shape[2]
                Imports.maps = Imports.maps[:,Imports.seq_start:Imports.seq_stop,:]
                Imports.maps = Imports.maps.reshape((Imports.nS, Imports.dim))
                # Update Hamming distances based on sequence cropping
                ref_short = Imports.ref_full[Imports.seq_start:Imports.seq_stop]
                if 'Hamming' in Imports.mave_col_names and Imports.combo_ref.currentText() != 'None':
                    hamming_new = []
                    for x in tqdm(Imports.mave['Sequence'], desc='Hamming update'):
                        hamming_new.append(round(spatial.distance.hamming(list(ref_short), list(x[Imports.seq_start:Imports.seq_stop])) * len(ref_short)))
                    Imports.mave['Hamming'] = hamming_new
                # Identify reference sequence (cropped) in mave dataset if present
                if Imports.combo_ref.currentText() == 'Custom':
                    Imports.ref_idx = ''
                    for idx, x in enumerate(Imports.mave['Sequence']):
                        if x[Imports.seq_start:Imports.seq_stop] == ref_short:
                            Imports.ref_idx = idx
                            break
                    if Imports.ref_idx == '':
                        print('No match to custom reference sequence (cropped) found in MAVE dataset.')
            else:
                Imports.map_crop = False
                Imports.maps = Imports.maps.reshape((Imports.nS, Imports.dim))
                if 'Hamming' in Imports.mave_col_names:
                    Imports.mave['Hamming'] = Imports.hamming_orig
                # Identify reference sequence in mave dataset if present
                if Imports.combo_ref.currentText() == 'Custom':
                    Imports.ref_idx = ''
                    for idx, x in enumerate(Imports.mave['Sequence']):
                        if x == Imports.ref_full:
                            Imports.ref_idx = idx
                            break
                    if Imports.ref_idx == '':
                        print('No match to custom reference sequence found in MAVE dataset.')

            if Imports.combo_ref.currentText() == 'None':
                Imports.ref_idx = ''

            # Position frequency matrix of background
            print('Calculating PFM of background...')
            seq_array_background = motifs.create(Imports.mave['Sequence'].str.slice(Imports.seq_start, Imports.seq_stop), alphabet=Imports.alphabet)
            Custom.pfm_background = seq_array_background.counts

            Custom.zord = list(range(Imports.nS))
            Custom.zord0 = Custom.zord

            Custom.logos_start = 0
            Custom.logos_stop = Imports.seq_length

            if 0: # TODO - is this needed?
                print('Calculating attribution errors of background...')
                Imports.maps_bg_on = True
                maps_background = Imports.maps.reshape((Imports.maps.shape[0], Imports.seq_length, Imports.maps_shape[2]))
                Imports.errors_background = np.linalg.norm(maps_background - np.mean(maps_background, axis=0), axis=(1,2))
            else:
                Imports.maps_bg_on = False

            tab2 = self.findChild(QtWidgets.QTabWidget).widget(2) # moved from within linkage_selected below
            tab2.ax.clear()

            if embedding_selected:
                def normalize(_d, to_sum=False, copy=True):
                    d = _d if not copy else np.copy(_d)
                    d -= np.min(d, axis=0)
                    d /= (np.sum(d, axis=0) if to_sum else np.ptp(d, axis=0))
                    return d
                
                Imports.embedding = np.load(Imports.embedding_fname)
                Imports.embedding = normalize(Imports.embedding)
                x = Imports.embedding[:,Custom.eig1_choice]
                y = Imports.embedding[:,Custom.eig2_choice]
                Custom.pts_orig = zip(x,y)
                Custom.pts_origX = x
                Custom.pts_origY = y

                # Create initial scatter plot
                Custom.scatter = Custom.ax.scatter(Custom.pts_origX[::Custom.plt_step],
                                                Custom.pts_origY[::Custom.plt_step],
                                                s=Custom.plt_marker,
                                                c=Imports.mave[Custom.cmap][::Custom.plt_step],
                                                cmap='jet', linewidth=Custom.plt_lw)

                # Set thin spine thickness for embedding plot
                for spine in Custom.ax.spines.values():
                    spine.set_linewidth(0.1)

                # Create initial colorbar
                divider = make_axes_locatable(Custom.ax)
                Custom.cax = divider.append_axes('right', size='5%', pad=0.05)
                Custom.cbar = Custom.figure.colorbar(Custom.scatter, cax=Custom.cax, orientation='vertical')
                Custom.cbar.ax.set_ylabel('DNN score', rotation=270, fontsize=6, labelpad=9)
                Custom.cbar.ax.tick_params(labelsize=6)

                Custom.entry_eig1.setMaximum(Imports.embedding.shape[1])
                Custom.entry_eig2.setMaximum(Imports.embedding.shape[1])
                Custom.entry_stepsize.setMaximum(Imports.nS-1)

                # Create cluster assignments using embedding clustering methods
                clusterer = Clusterer(Imports.maps, gpu=False)
                
                # Get clustering method and parameters from UI
                clustering_method = Imports.combo_embedding_method.currentText()
                n_clusters = Imports.spin_embedding_clusters.value()
                
                if clustering_method == 'kmeans':
                    cluster_assignments = clusterer.cluster(
                        embedding=Imports.embedding, 
                        method='kmeans', 
                        n_clusters=n_clusters
                    )
                elif clustering_method == 'dbscan':
                    cluster_assignments = clusterer.cluster(
                        embedding=Imports.embedding, 
                        method='dbscan'
                    )
  
                Imports.clusters = pd.DataFrame({'Cluster': cluster_assignments})
                Imports.cluster_col = 'Cluster'
                        
                # Preprocess all cluster logos
                print("Preprocessing cluster logos...")
                Imports.batch_logo_instances.clear()  # Clear any existing instances
                
                # Get unique clusters
                unique_clusters = Imports.clusters[Imports.cluster_col].unique()

                # Use first two dimensions of embedding for visualization
                x = Imports.embedding[:,0]
                y = Imports.embedding[:,1]

                tab2.embedding = np.array([x, y]).T
                tab2.pts_orig = zip(x,y)
                tab2.pts_origX = x
                tab2.pts_origY = y

                tab2.clusters_idx = np.arange(Imports.clusters[Imports.cluster_col].max()+1) # TODO: is this needed below or use Predef version?
                Predef.clusters_idx = np.arange(Imports.clusters[Imports.cluster_col].max()+1)
                tab2.cluster_sorted_indices = np.arange(Imports.clusters[Imports.cluster_col].max()+1)
                Imports.mave['Cluster'] = Imports.clusters['Cluster']

                tab2.cluster_colors = Imports.clusters[Imports.cluster_col].values[::Custom.plt_step]
                #if Imports.cluster_col == 'Cluster_sort': # TODO - is this still needed?
                #    unique_clusters = np.unique(tab2.cluster_colors)
                #    randomized_mapping = {cluster: i for i, cluster in enumerate(np.random.permutation(unique_clusters))}
                #    tab2.cluster_colors = [randomized_mapping[cluster] for cluster in tab2.cluster_colors]

                tab2.scatter = tab2.ax.scatter(tab2.pts_origX[::Custom.plt_step],
                                               tab2.pts_origY[::Custom.plt_step],
                                               s=Custom.plt_marker,
                                               c=tab2.cluster_colors,
                                               cmap='tab10',
                                               linewidth=Custom.plt_lw)

                # Set thin spine thickness for embedding plot
                for spine in tab2.ax.spines.values():
                    spine.set_linewidth(0.1)

                #tab2.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % int(Imports.psi_1+1), fontsize=6) TODO - weird cropping
                #tab2.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % int(Imports.psi_2+1), fontsize=6) TODO
                tab2.ax.tick_params(axis='both', which='major', labelsize=4)
                tab2.ax.get_xaxis().set_ticks([])
                tab2.ax.get_yaxis().set_ticks([])
                tab2.ax.set_title('Click on a cluster to view its logo (currently showing cluster 0)', fontsize=5)
                tab2.ax.autoscale()
                tab2.ax.margins(x=0)
                tab2.ax.margins(y=0)
                tab2.canvas.draw()

            elif linkage_selected:
                # Create cluster assignments using Clusterer
                clusterer = Clusterer(Imports.maps, gpu=False)
                # Use values from UI instead of hardcoding
                param = Imports.spin_cut_param.value()
                criterion = Imports.combo_cut_criterion.currentText().lower()
                cluster_assignments, _ = clusterer.get_cluster_labels(Imports.linkage, criterion=criterion, n_clusters=param)
                if criterion == 'maxclust':
                    cluster_assignments -= 1 # fcluster is 1-based, convert to 0-based
                Imports.clusters = pd.DataFrame({'Cluster': cluster_assignments})
                Imports.cluster_col = 'Cluster'
                        
                # Preprocess all cluster logos
                print("Preprocessing cluster logos...")
                Imports.batch_logo_instances.clear()  # Clear any existing instances
                
                # Get unique clusters
                unique_clusters = Imports.clusters[Imports.cluster_col].unique()

                # --- Dendrogram Implementation ---
                tab2.ax.clear()

                # Use the existing plot_dendrogram function from Clusterer
                # Always truncate dendrograms
                
                # Calculate cut_level for dendrogram display
                cut_level = None
                if criterion == 'maxclust':
                    n_clusters = param
                    cut_level = Imports.linkage[Imports.nS - n_clusters - 1, 2]
                else: # distance
                    cut_level = param
                
                clusterer.plot_dendrogram(Imports.linkage, ax=tab2.ax, cut_level=cut_level, truncate=True, cut_level_truncate=cut_level, criterion=criterion, n_clusters=param if criterion == 'maxclust' else None, gui=True)
                
                # Setup coordinates for clickable leaves
                with plt.ioff():
                    fig_temp, ax_temp = plt.subplots()
                    R = hierarchy.dendrogram(Imports.linkage, ax=ax_temp, no_plot=True)
                    plt.close(fig_temp)
                
                n_samples = Imports.nS
                x_coords = np.zeros(n_samples)
                y_coords = np.zeros(n_samples) # Leaves are at y=0

                # Map leaf order to x-coordinates
                for i, leaf_idx in enumerate(R['leaves']):
                    x_coords[leaf_idx] = 10 * i + 5
                
                tab2.pts_origX = x_coords
                tab2.pts_origY = y_coords
                tab2.embedding = np.column_stack([x_coords, y_coords])
                tab2.cluster_colors = cluster_assignments # Store for onclick highlighting

                # Adjust plot appearance
                tab2.ax.tick_params(axis='y', which='major', labelsize=4)
                tab2.ax.set_title('Dendrogram-based clustering (click on a leaf to view its cluster logo)', fontsize=5)

                tab2.clusters_idx = np.arange(Imports.clusters[Imports.cluster_col].max()+1)
                Predef.clusters_idx = np.arange(Imports.clusters[Imports.cluster_col].max()+1)
                tab2.cluster_sorted_indices = np.arange(Imports.clusters[Imports.cluster_col].max()+1)
                Imports.mave['Cluster'] = Imports.clusters['Cluster']
            
            # Store original shape and reshape maps to 3D for logo processing
            original_shape = Imports.maps.shape
            if len(Imports.maps.shape) == 2:
                N = Imports.maps.shape[0]
                L = Imports.seq_length
                C = Imports.maps_shape[2] if hasattr(Imports, 'maps_shape') else len(Imports.alphabet)
                Imports.maps = Imports.maps.reshape(N, L, C)
            
            # Create Clusterer instance
            clusterer = Clusterer(
                attribution_maps=Imports.maps,
                gpu=False
            )
            clusterer.cluster_labels = Imports.clusters[Imports.cluster_col].values
            
            # Create MetaExplainer instance
            if Imports.cluster_sort_method == 'Median activity':
                meta_sort_method = 'median'
            else:  # No reordering
                meta_sort_method = None
            
            meta = MetaExplainer(
                clusterer=clusterer,
                mave_df=Imports.mave,
                attributions=Imports.maps,
                ref_idx=Imports.ref_idx,
                background_separation=Imports.checkbox_background_separation.isChecked(),
                mut_rate=Imports.spin_mutation_rate.value(),
                sort_method=meta_sort_method,
                alphabet=Imports.alphabet
            )
            
            # Generate logos for all clusters at once (always standard logos)
            meta_logos = meta.generate_logos(
                logo_type=Imports.batch_logo_type,
                background_separation=False,  # Always generate standard logos first
                mut_rate=Imports.spin_mutation_rate.value(),
                entropy_multiplier=Imports.spin_entropy_multiplier.value(),
                adaptive_background_scaling=Imports.checkbox_adaptive_scaling.isChecked(),
                figsize=(10, 2.5)
            )
            
            # Store MetaExplainer instance and generated logos
            Imports.meta_explainer = meta
            Imports.meta_logos = meta_logos
            
            # Update cluster assignments to match MetaExplainer's sorting
            if meta.cluster_order is not None:
                # Create mapping from original cluster indices to sorted positions
                cluster_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(meta.cluster_order)}
                # Store reverse mapping for Cell class lookup (original -> remapped)
                Imports.cluster_reverse_mapping = cluster_mapping
                # Remap cluster assignments
                Imports.clusters[Imports.cluster_col] = Imports.clusters[Imports.cluster_col].map(cluster_mapping)
                # Update mave DataFrame cluster assignments
                Imports.mave['Cluster'] = Imports.mave['Cluster'].map(cluster_mapping)
            else:
                # No sorting, so no mapping needed
                Imports.cluster_reverse_mapping = None
            
            # Create BatchLogo instances for each cluster from meta logos
            for cluster_idx in range(len(unique_clusters)):
                # Always create standard logos
                logo_key = (cluster_idx, Imports.batch_logo_type)
                batch_logo = BatchLogo(
                    values=meta_logos.values[cluster_idx:cluster_idx+1],
                    alphabet=Imports.alphabet,
                    figsize=(10, 2.5),
                    batch_size=1,
                    font_name='Arial Rounded MT Bold',
                    fade_below=0.5,
                    shade_below=0.5,
                    width=0.9,
                    center_values=True,
                    show_progress=False
                )
                batch_logo.process_all()
                Imports.batch_logo_instances[logo_key] = batch_logo
            
            # If background separation is enabled, also create background-separated versions for all clusters
            if Imports.checkbox_background_separation.isChecked():
                print("Generating background-separated logos...")
                # Create a separate MetaExplainer instance for background-separated logos
                meta_bg = MetaExplainer(
                    clusterer=clusterer,
                    mave_df=Imports.mave,
                    attributions=Imports.maps,
                    ref_idx=Imports.ref_idx,
                    background_separation=True,
                    mut_rate=Imports.spin_mutation_rate.value(),
                    sort_method=meta_sort_method,
                    alphabet=Imports.alphabet
                )
                # Generate background-separated logos for all clusters at once
                meta_logos_separated = meta_bg.generate_logos(
                    logo_type=Imports.batch_logo_type,
                    background_separation=True,
                    mut_rate=Imports.spin_mutation_rate.value(),
                    entropy_multiplier=Imports.spin_entropy_multiplier.value(),
                    adaptive_background_scaling=Imports.checkbox_adaptive_scaling.isChecked(),
                    figsize=(10, 2.5)
                )
                
                # Create BatchLogo instances for background-separated versions
                for cluster_idx in range(len(unique_clusters)):
                    logo_key_separated = (cluster_idx, f'{Imports.batch_logo_type}_separated')
                    batch_logo_separated = BatchLogo(
                        values=meta_logos_separated.values[cluster_idx:cluster_idx+1],
                        alphabet=Imports.alphabet,
                        figsize=(10, 2.5),
                        batch_size=1,
                        font_name='Arial Rounded MT Bold',
                        fade_below=0.5,
                        shade_below=0.5,
                        width=0.9,
                        center_values=True,
                        show_progress=False
                    )
                    batch_logo_separated.process_all()
                    Imports.batch_logo_instances[logo_key_separated] = batch_logo_separated
            
            # Create variability logo (shows all clusters overlaid) - AFTER background separation logic
            print("Generating variability logo...")
            if Imports.checkbox_background_separation.isChecked():
                # Use background-separated data for variability logo
                variability_logo = BatchLogo(
                    values=meta_logos_separated.values,  # All clusters with background separation
                    alphabet=Imports.alphabet,
                    figsize=(10, 2.5),
                    batch_size=len(unique_clusters),
                    font_name='Arial Rounded MT Bold',
                    fade_below=0.5,
                    shade_below=0.5,
                    width=0.9,
                    center_values=True,
                    show_progress=False
                )
            else:
                # Use standard data for variability logo
                variability_logo = BatchLogo(
                    values=meta_logos.values,  # All clusters
                    alphabet=Imports.alphabet,
                    figsize=(10, 2.5),
                    batch_size=len(unique_clusters),
                    font_name='Arial Rounded MT Bold',
                    fade_below=0.5,
                    shade_below=0.5,
                    width=0.9,
                    center_values=True,
                    show_progress=False
                )
            
            variability_logo.process_all()
            Imports.batch_logo_instances['variability'] = variability_logo
            
            # Pre-compute the variability logo figure for instant display
            print("Pre-computing variability logo figure...")
            variability_fig, variability_ax = variability_logo.draw_variability_logo(view_window=None, figsize=(20, 1.5), border=False)
            Imports.variability_figure = variability_fig
            Imports.variability_axis = variability_ax

            # Create average background logo only if background separation is enabled
            if Imports.checkbox_background_separation.isChecked():
                print("Generating average background logo...")
                # Get the background data from the meta_bg object (which has background_separation=True)
                # The background attribute is available after background separation is computed
                if hasattr(meta_bg, 'background') and meta_bg.background is not None:
                    average_background_logo = BatchLogo(
                        values=meta_bg.background[np.newaxis, :, :],
                        alphabet=Imports.alphabet,
                        figsize=(10, 2.5),
                        batch_size=1,
                        font_name='Arial Rounded MT Bold',
                        fade_below=0.5,
                        shade_below=0.5,
                        width=0.9,
                        center_values=True,
                        show_progress=False
                    )
                    average_background_logo.process_all()
                    Imports.batch_logo_instances['average_background'] = average_background_logo
                else:
                    print("Warning: No background data available for average background logo")

                            
            # Calculate global y-axis limits for fixed scaling
            y_mins = []
            y_maxs = []
            
            for cluster_idx in range(len(unique_clusters)):
                logo_key = (cluster_idx, Imports.batch_logo_type)
                if logo_key in Imports.batch_logo_instances:
                    batch_logo = Imports.batch_logo_instances[logo_key]
                    matrix = batch_logo.values[0]  # Shape: (seq_length, alphabet_size)
                    
                    # Calculate positive and negative sums at each position
                    positive_mask = matrix > 0
                    positive_matrix = matrix * positive_mask
                    positive_sums = positive_matrix.sum(axis=1)
                    
                    negative_mask = matrix < 0
                    negative_matrix = matrix * negative_mask
                    negative_sums = negative_matrix.sum(axis=1)
                    
                    y_mins.append(negative_sums.min())
                    y_maxs.append(positive_sums.max())
            
            if y_mins and y_maxs:
                Imports.global_y_min = min(y_mins)
                Imports.global_y_max = max(y_maxs)
            else: # TODO - will this ever happen?
                Imports.global_y_min = -1.0
                Imports.global_y_max = 1.0
            
            # Calculate global y-axis limits for background-separated logos if they exist
            if Imports.checkbox_background_separation.isChecked():
                bg_y_mins = []
                bg_y_maxs = []
                
                for cluster_idx in range(len(unique_clusters)):
                    logo_key_separated = (cluster_idx, f'{Imports.batch_logo_type}_separated')
                    if logo_key_separated in Imports.batch_logo_instances:
                        batch_logo_separated = Imports.batch_logo_instances[logo_key_separated]
                        matrix = batch_logo_separated.values[0]  # Shape: (seq_length, alphabet_size)
                        
                        # Calculate positive and negative sums at each position
                        positive_mask = matrix > 0
                        positive_matrix = matrix * positive_mask
                        positive_sums = positive_matrix.sum(axis=1)
                        
                        negative_mask = matrix < 0
                        negative_matrix = matrix * negative_mask
                        negative_sums = negative_matrix.sum(axis=1)
                        
                        bg_y_mins.append(negative_sums.min())
                        bg_y_maxs.append(positive_sums.max())
                
                if bg_y_mins and bg_y_maxs:
                    Imports.global_y_min_separated = min(bg_y_mins)
                    Imports.global_y_max_separated = max(bg_y_maxs)
                else:
                    Imports.global_y_min_separated = -1.0
                    Imports.global_y_max_separated = 1.0
            
            # Generate Mechanism Summary Matrix (MSM) for AllStats
            print("Generating Mechanism Summary Matrix...")
            Predef.df = meta.generate_msm(gpu=False)
            
            # If "No reordering" is selected, preserve natural cluster order
            if Imports.cluster_sort_method == 'No reordering':
                # Get the natural order of clusters as they appear in the data
                natural_order = pd.Series(Imports.clusters['Cluster'].values).drop_duplicates().values
                # Create a mapping from cluster to desired position
                order_mapping = {cluster: idx for idx, cluster in enumerate(natural_order)}
                # Add a temporary column for sorting
                Predef.df['_sort_order'] = Predef.df['Cluster'].map(order_mapping)
                # Sort by this temporary column
                Predef.df = Predef.df.sort_values('_sort_order').drop('_sort_order', axis=1)
                            
            # Always reshape maps back to original shape, even if an error occurs
            Imports.maps = Imports.maps.reshape(original_shape)
            # Clean up any remaining figures
            plt.close('all')

            # Enable appropriate tabs based on what's loaded
            if embedding_selected:
                tabs.setTabEnabled(1, True)  # Custom Clusters tab
                tabs.setTabEnabled(2, True)  # Predefined Clusters tab
                tabs.setCurrentIndex(2)
                Custom.btn_reset.click()
            elif linkage_selected:
                tabs.setTabEnabled(1, False)  # Custom Clusters tab
                tabs.setTabEnabled(2, True)   # Predefined Clusters tab
                tabs.setCurrentIndex(2)
                
            tab2.choose_cluster.setMaximum(Imports.clusters[Imports.cluster_col].max())
            tab2.choose_cluster.setSuffix(' / %s' % Imports.clusters[Imports.cluster_col].max())

            # Initialize logo display with blank image
            tab2.k_idxs = ''
            # Show initial logo for cluster 0
            tab2.update_logo()
            # Set cluster 0 in the spinbox and trigger the View button
            tab2.choose_cluster.setValue(0)

            embedding_selected = Imports.checkbox_embedding.isChecked() and Imports.embedding_fname != ''
            tab2.highlight_cluster()
            
            # Show/hide background separation checkbox based on P1 setting
            if Imports.checkbox_background_separation.isChecked():
                Predef.checkbox_bg_separation.setVisible(True)
                Predef.label_bg_separation.setVisible(True)
                Predef.checkbox_avg_background.setVisible(True)
                Predef.label_avg_background.setVisible(True)
            else:
                Predef.checkbox_bg_separation.setVisible(False)
                Predef.label_bg_separation.setVisible(False)
                Predef.checkbox_bg_separation.setChecked(False)  # Ensure it's unchecked when hidden
                Predef.checkbox_avg_background.setVisible(False)
                Predef.label_avg_background.setVisible(False)
                Predef.checkbox_avg_background.setChecked(False)  # Ensure it's unchecked when hidden
            
            # Update embedding plot to reflect cluster 0 is selected
            tab2.canvas.draw()

            print('Processing complete.')

        else:
            box = QMessageBox(self)
            box.setWindowTitle('%s Error' % progname)
            box.setText('<b>Input Error</b>')
            box.setFont(font_standard)
            box.setIcon(QMessageBox.Information)
            box.setInformativeText('All entries must be complete.')
            box.setStandardButtons(QMessageBox.Ok)
            box.setDefaultButton(QMessageBox.Ok)
            ret = box.exec_()

        
    def closeEvent(self, ce): # safety message if user clicks to exit via window button
        msg = "<span style='font-weight:normal;'>\
               Performing this action will close the program.\
               <br /><br />\
               Do you want to proceed?\
               </span>"
        box = QMessageBox(self)
        box.setWindowTitle('%s Warning' % progname)
        box.setText('<b>Exit Warning</b>')
        box.setFont(font_standard)
        box.setIcon(QMessageBox.Warning)
        box.setInformativeText(msg)
        box.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
        reply = box.exec_()
        if reply == QMessageBox.Yes:
            self.close()
            sys.exit()
        else:
            ce.ignore()

    def fileRestart(self):
        msg = "<span style='font-weight:normal;'>\
               Performing this action will restart the program and reset all user inputs.\
               <br /><br />\
               Do you want to proceed?\
               </span>"
        box = QMessageBox(self)
        box.setWindowTitle('%s Warning' % progname)
        box.setText('<b>Restart Warning</b>')
        box.setFont(font_standard)
        box.setIcon(QMessageBox.Warning)
        box.setInformativeText(msg)
        box.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
        reply = box.exec_()
        if reply == QMessageBox.Yes:
            try:
                p = psutil.Process(os.getpid())
                for handler in p.open_files() + p.connections():
                    os.close(handler.fd)
            except Exception as e:
                logging.error(e)

            python = sys.executable
            os.execl(python, python, * sys.argv)
        else:
            pass

    def about(self):
        box = QMessageBox(self)
        box.setWindowTitle('%s About' % progname)
        box.setText('<b>%s</b>' % progname)
        box.setFont(font_standard)
        p1 = QtGui.QPixmap(os.path.join(icon_dir, '256x256.png'))
        box.setIconPixmap(QtGui.QPixmap(p1))
        p2 = p1.scaled(150,150)
        box.setIconPixmap(p2)
        box.setInformativeText('<span style="font-weight:normal;">\
                                Developed by Evan E Seitz\
                                <br />CSHL, 2023-2025\
                                <br /><br />\
                                <b>LICENSE:</b>\
                                <br />\
                                You should have received a copy of the GNU General Public\
                                License along with this program.\
                                <br /><br />\
                                <b>CONTACT:</b>\
                                <br />\
                                Please refer to our repository for all inquiries, including\
                                preferred methods for contacting our team for software support.\
                                <br /><br />\
                                </span>')
        box.setStandardButtons(QMessageBox.Ok)        
        ret = box.exec_()

    # TODO - ensure that DNN, Hamming, GIAtext automatically propagated on tab 2 based on what's in the data file

    def show_import_help(self):
        """Show help dialog for Import Files tab"""
        help_text = """
        <style>
            .subsection {
                margin-left: 20px;
                border-left: 3px solid #ccc;
                padding-left: 10px;
            }
        </style>
        <h3>Import Files Tab Guide</h3>
        
        <p>This tab allows you to import and prepare your data for SEAM analysis. Here's what each option does:</p>

        <h4>1. Data File (*.csv)</h4>
        <p>Import your sequence library and measurements:</p>
        <ul>
            <li><b>Required columns:</b> 'Sequence' (DNA/RNA sequences) and at least one measurement column (e.g., 'DNN' for Deep Neural Network predictions; 'Hamming' for Hamming distance to reference sequence; 'GIA' for GIA score)</li>
            <li>Each row represents one sequence and its associated measurements</li>
            <li>See SQUID Mutagenizer class for generating in silico sequence libraries</li>
            <li>See SEAM Compiler class for converting sequences and measurements to the required format</li>

        </ul>

        <div class="subsection">
            <h4>Reference Sequence</h4>
            <p>Choose how to handle the reference sequence:</p>
            <ul>
                <li><b>First row:</b> Uses the first sequence in your data file as reference (default if generated with SQUID Mutagenizer class)</li>
                <li><b>Custom:</b> Enter your own reference sequence</li>
                <li><b>None:</b> Disable analysis with respect to reference sequence (e.g., for global libraries)</li>
            </ul>
        </div>

        <h4>2. Attribution Maps (*.npy, *.npz)</h4>
        <p>Import your attribution maps:</p>
        <ul>
            <li>Shape should be (N, L, A) where:</li>
            <li>N = number of sequences</li>
            <li>L = sequence length</li>
            <li>A = alphabet size (e.g., 4 for DNA)</li>
            <li>Attribution maps quantify the importance of each position/base</li>
            <li>See SEAM Attributer class for generating attribution maps from sequence library using various attribution methods</li>
        </ul>

        <div class="subsection">
            <h4>Sequence Range</h4>
            <p>Select which portion of sequences to analyze:</p>
            <ul>
                <li>Use start/stop boxes to crop sequences if attribution maps were cropped with respect to the sequences</li>
                <li>Helps focus analysis on specific regions, granting higher resolution of clustered features</li>
                <li>Choice must match the cropped boundaries of attribution maps which were used to generate clusters/linkage</li>
            </ul>
        </div>

        <h4>3. Clustering Options</h4>
        <p>Choose how to cluster your sequences. You must select either Embedding OR Linkage (but not both):</p>
        
        <div class="subsection">
            <h4>Embedding (*.npy)</h4>
            <ul>
                <li>Import pre-computed embedding of attribution maps</li>
                <li>Shape should be (N, Z) where Z is number of dimensions</li>
                <li>Used for visualization in Custom Clusters and Predefined Clusters tabs</li>
                <li><b>Clustering method:</b> Choose between kmeans (specify number of clusters) or dbscan</li>
                <li><b>Number of clusters:</b> Specify number of clusters for kmeans clustering</li>
                <li><b>Cluster sorting:</b> Median activity (default) or no reordering</li>
                <li>See SEAM Clusterer class for generating embedding of attribution maps using dimensionality reduction (e.g., UMAP, t-SNE, PCA)</li>
            </ul>
            
            <h4>Linkage (*.npy)</h4>
            <ul>
                <li>Import pre-computed hierarchical clustering linkage matrix</li>
                <li>Shape should be (N-1, 4) where N is number of sequences (e.g., see scipy.cluster.hierarchy.linkage)</li>
                <li>Used for visualization in Predefined Clusters tab</li>
                <li><b>Cut criterion:</b> maxclust (specify number of clusters) or distance (specify distance threshold)</li>
                <li><b>Number of clusters:</b> If maxclust, specify number of clusters for maxclust clustering</li>
                <li><b>Distance threshold:</b> If distance, specify distance threshold for hierarchical clustering</li>
                <li><b>Cluster sorting:</b> Median activity (default) or no reordering</li>
                <li>See SEAM Clusterer class for generating linkage matrix from attribution maps (hierarchical clustering)</li>
            </ul>
        </div>

        <h4>4. Additional Options</h4>
        <div class="subsection">
            <h4>Background Separation</h4>
            <p>Enable for analysis of local sequence libraries:</p>
            <ul>
                <li><b>Mutation rate:</b> Set the mutation rate for analysis (0.0 to 1.0, default: 0.10)</li>
                <li><b>Adaptive scaling:</b> Enable cluster-specific background scaling for better separation (default: enabled)</li>
                <li><b>Entropy multiplier:</b> Control background position identification stringency (0.1 to 1.0, default: 0.5). Lower values are more stringent.</li>
            </ul>
        </div>

        <p>After importing your data, click "Confirm Imports" at the bottom of the window to proceed to analysis</p>
        """
        
        dialog = HelpDialog('Import Files Help', help_text, self)
        dialog.exec_()

    def show_custom_help(self):
        """Show help dialog for Custom Clusters tab"""
        help_text = """
        <style>
            .subsection {
                margin-left: 20px;
                border-left: 3px solid #ccc;
                padding-left: 10px;
            }
        </style>
        <h3>Custom Clusters Tab Guide</h3>
        
        <p>This tab allows you to interactively explore and manually select clusters in your embedding space. You can draw custom regions around groups of sequences to analyze their shared characteristics.</p>

        <h4>Prerequisites</h4>
        <p><b>Required:</b> You must have loaded data with an embedding file (not linkage) on the Import Files tab. The Custom Clusters tab is only available when using embedding-based clustering.</p>

        <h4>Main Interface</h4>
        <p>The main area shows a 2D scatter plot of your sequences in embedding space:</p>
        <ul>
            <li><b>Points:</b> Each point represents a sequence from your library</li>
            <li><b>Colors:</b> Points are colored by measurement values (DNN score, GIA score, Hamming distance, etc.)</li>
            <li><b>Axes:</b> X and Y coordinates represent different dimensions of your embedding (Ψ1, Ψ2, etc.)</li>
        </ul>

        <h4>Interactive Cluster Selection</h4>
        <ol>
            <li><b>Draw a polygon:</b> Click on the plot to place points that will form a polygon around sequences of interest</li>
            <li><b>Connect the path:</b> Click "Connect Path" to close the polygon and identify sequences within it</li>
            <li><b>View cluster:</b> Click "View Cluster" to analyze the selected sequences</li>
        </ol>

        <h4>Display Controls</h4>
        <div class="subsection">
            <h4>Coordinate Selection</h4>
            <ul>
                <li><b>Ψ1:</b> Select which embedding dimension to display on the X-axis</li>
                <li><b>Ψ2:</b> Select which embedding dimension to display on the Y-axis</li>
            </ul>
            
            <h4>Visualization Options</h4>
            <ul>
                <li><b>Color map:</b> Choose what to color points by (DNN, GIA, Hamming, Task, or Histogram)</li>
                <li><b>Theme:</b> Light or dark theme for the plot</li>
                <li><b>Drawing order:</b> Original, ascending, or descending order for point rendering</li>
                <li><b>Skip every:</b> Reduce point density for better performance with large datasets</li>
                <li><b>Marker size:</b> Adjust the size of points in the scatter plot</li>
                <li><b>Plot reference:</b> Highlight the reference sequence (if available) with a star marker</li>
            </ul>
        </div>

        <h4>Cluster Analysis</h4>
        <p>When you select a cluster and click "View Cluster", a new window opens showing:</p>
        <ul>
            <li><b>Attribution logo:</b> Average attribution map for sequences in the cluster</li>
            <li><b>Sequence logo:</b> Position frequency matrix showing sequence patterns</li>
            <li><b>Statistics:</b> Detailed analysis of the cluster's characteristics</li>
            <li><b>Sequence table:</b> View all individual sequences in the cluster</li>
        </ul>

        <h4>Tips</h4>
        <ul>
            <li>Use the zoom and pan tools to explore different regions of your embedding</li>
            <li>Use the "Reset Path" button to start over with a new polygon</li>
            <li>For large datasets, increase the "Skip every" value to improve performance</li>
        </ul>
        """
        dialog = HelpDialog('Custom Clusters Help', help_text, self)
        dialog.exec_()

    def show_predefined_help(self):
        """Show help dialog for Predefined Clusters tab"""
        help_text = """
        <style>
            .subsection {
                margin-left: 20px;
                border-left: 3px solid #ccc;
                padding-left: 10px;
            }
        </style>
        <h3>Predefined Clusters Tab Guide</h3>
        
        <p>This tab provides automated cluster analysis and visualization based on your imported clustering data. The interface and functionality differ depending on whether you imported an embedding or linkage file on the Import Files tab.</p>

        <h4>Prerequisites</h4>
        <p><b>Required:</b> You must have loaded data with either an embedding file OR a linkage file (but not both) on the Import Files tab. The Predefined Clusters tab becomes available after confirming your imports.</p>

        <h4>Two Different Modes</h4>
        
        <div class="subsection">
            <h4>Embedding Mode (Interactive)</h4>
            <p>When you imported an embedding file, this tab provides an interactive scatter plot:</p>
            <ul>
                <li><b>Clickable clusters:</b> Click directly on any cluster in the embedding space to view its analysis</li>
                <li><b>Real-time updates:</b> Logos and statistics update immediately when you click on different clusters</li>
                <li><b>Visual feedback:</b> The selected cluster is highlighted in the plot</li>
                <li><b>Cluster navigation:</b> Use the "Choose cluster" spinner to jump to specific cluster numbers</li>
            </ul>
            
            <h4>Linkage Mode (Dendrogram View)</h4>
            <p>When you imported a linkage file, this tab shows a hierarchical clustering dendrogram:</p>
            <ul>
                <li><b>Dendrogram visualization:</b> Displays the hierarchical structure of your clusters</li>
                <li><b>Cluster selection:</b> Use the "Choose cluster" spinner to select clusters (interactive clicking coming in future updates)</li>
                <li><b>Truncated view:</b> The dendrogram is truncated to show only the clusters specified by your maxclust/distance parameters from the Import Files tab</li>
            </ul>
        </div>

        <h4>Main Interface Components</h4>
        
        <div class="subsection">
            <h4>Visualization Area</h4>
            <ul>
                <li><b>Top plot:</b> Shows either the embedding scatter plot (embedding mode) or dendrogram (linkage mode)</li>
                <li><b>Bottom logo:</b> Displays the cluster-averaged attribution logo for the currently selected cluster</li>
            </ul>
            
            <h4>Cluster Selection</h4>
            <ul>
                <li><b>Choose cluster:</b> Use the spinner to select a specific cluster by number</li>
            </ul>
            
            <h4>Logo Display Options</h4>
            <ul>
                <li><b>Logo visualization scheme:</b> Choose between "Average of maps" (attribution-based) or sequence-based options</li>
                <li><b>Y-axis scaling:</b> Adaptive (auto-scaled per logo) or Fixed (consistent scale across logos)</li>
                <li><b>Background separation:</b> When enabled, shows background-separated logos (removes background signal)</li>
                <li><b>Variability logo:</b> Shows a static logo representing sequence variability across all clusters (unchanging between cluster selections). If background separation is enabled on the Imports tab, this logo will correspond to background-separated cluster-averaged logos.</li>
                <li><b>Average background:</b> Shows a static logo representing the average background signal across all clusters (unchanging between cluster selections). This option is only available when background separation analysis is chosen on the Imports tab.</li>
            </ul>
        </div>

        <h4>Analysis Tools</h4>
        
        <div class="subsection">
            <h4>Cluster Analysis</h4>
            <ul>
                <li><b>Clustered Sequences:</b> View all individual sequences in the current cluster in a table format</li>
                <li><b>Intra-cluster Statistics:</b> Detailed statistical analysis of the selected cluster</li>
                <li><b>Inter-cluster Statistics:</b> Compare the selected cluster with all other clusters using box plots or bar charts</li>
                <li><b>Cluster Summary Matrix (CSM):</b> View a heatmap showing cluster characteristics across all positions</li>
                <ul>
                    <li><b>Metrics:</b> Choose between positional Shannon entropy, percent mismatches to reference, or consensus per cluster</li>
                    <li><b>Marginal distributions:</b> Analyze the distribution of cluster characteristics across positions and clusters</li>
                </ul>
            </ul>
        </div>

        <h4>Workflow Tips</h4>
        <ul>
            <li><b>Start with overview:</b> Use the Cluster Summary Matrix to get a global view of your clusters</li>
            <li><b>Explore interesting clusters:</b> Click on clusters (embedding mode) or use the spinner to explore clusters with interesting patterns</li>
            <li><b>Compare clusters:</b> Use Inter-cluster Statistics to understand how clusters differ from each other</li>
            <li><b>Zoom for details:</b> Use the matplotlib zoom tool to examine specific regions of the visualization</li>
            <li><b>Adjust logo display:</b> Try different y-axis scaling options to better visualize your data</li>
        </ul>

        <h4>Key Differences from Custom Clusters</h4>
        <ul>
            <li><b>Automated clustering:</b> Clusters are pre-defined based on your imported clustering data</li>
            <li><b>Systematic analysis:</b> Easy to systematically explore all clusters rather than manually drawing regions</li>
            <li><b>Global statistics:</b> Access to cluster summary matrix and intra- and inter-cluster comparisons</li>
            <li><b>Reproducible results:</b> Same clusters will always be available, unlike manually drawn regions</li>
        </ul>
        """
        dialog = HelpDialog('Predefined Clusters Help', help_text, self)
        dialog.exec_()


class HelpDialog(QtWidgets.QDialog):
    def __init__(self, title, text, parent=None):
        super(HelpDialog, self).__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        
        # Create layout
        layout = QtWidgets.QVBoxLayout()
        
        # Create text browser widget for rich text display
        text_browser = QtWidgets.QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        text_browser.setHtml(text)
        text_browser.setMinimumWidth(800)
        text_browser.setMinimumHeight(600)
        
        # Add text browser to layout
        layout.addWidget(text_browser)
        
        # Add OK button
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)
        
        # Set the layout
        self.setLayout(layout)



if __name__ == '__main__':
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    QtCore.QCoreApplication.setApplicationName(progname)
    QApplication.setStyle(QStyleFactory.create('Fusion'))
    
    # Force light theme
    app.setStyleSheet("""
        QWidget { background-color: white; color: black; }
        QLineEdit:disabled { background-color: #f0f0f0; color: #808080; }
        QComboBox:disabled { background-color: #f0f0f0; color: #808080; }
        QSpinBox:disabled { background-color: #f0f0f0; color: #808080; }
        QDoubleSpinBox:disabled { background-color: #f0f0f0; color: #808080; }
        QPushButton:disabled { background-color: #f0f0f0; color: #808080; }
        QCheckBox:disabled { color: #808080; }
        QTableWidget:disabled { background-color: #f0f0f0; color: #808080; }
        QMainWindow:disabled { background-color: #f0f0f0; color: #808080; }
        QDialog:disabled { background-color: #f0f0f0; color: #808080; }
        QTextBrowser:disabled { background-color: #f0f0f0; color: #808080; }
        QDialogButtonBox:disabled { background-color: #f0f0f0; color: #808080; }
        QTabBar:disabled { background-color: #f0f0f0; color: #808080; }
        QTabBar::tab:disabled { background-color: #f0f0f0; color: #808080; }
    """) # Add disabled state styling for all common widgets
        
    # Set app icon for tray
    app_icon = QtGui.QIcon()
    app_icon.addFile(os.path.join(icon_dir, '256x256.png'), QtCore.QSize(256,256))
    app.setWindowIcon(app_icon)
        
    w = MainWindow()
    w.setWindowTitle('%s' % progname)
    w.show()
    sys.exit(app.exec_())