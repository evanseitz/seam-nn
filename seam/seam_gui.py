'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright Evan Seitz 2023-2025
Cold Spring Harbor Laboratory
Contact: evan.e.seitz@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''


import sys, os
sys.dont_write_bytecode = True
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator # integer ticks
from pylab import loadtxt
import pandas as pd
import logomaker
from Bio import motifs # 'pip install biopython'
from scipy import spatial # 'pip install scipy'
from scipy.spatial import distance
from scipy.cluster import hierarchy
import seaborn as sns # pip install seaborn
import squid.utils as squid_utils # 'pip install squid-nn'
from tqdm import tqdm
def warn(*args, **kwargs): # ignore matplotlib deprecation warnings:
    pass
import warnings
warnings.warn = warn


try:
    import impress
except ImportError:
    pass

try:
    import utilities
except ImportError:
    pass


# minimum installation:
    # conda create --name seam python==3.7.2 ### new: 3.11
    # pip install --upgrade pip
    # pip install squid-nn
    # pip install logomaker
    # pip install PyQt5
    # pip3 install --user psutil
    # pip install biopython
    # pip install tensorflow==2.11.0 #### new: no version
    # pip install scipy
    # pip install seaborn
# also:
    # pip install -U scikit-learn
    #   pip install scikit-network
    #   ##pip install pynndescent==0.5.8 #don't use
    #   ##pip install umap-learn #don't use
    # conda install -c conda-forge umap-learn
    # pip install pysam
    # pip install matplotlib==3.6 #at the end
    #   pip install statsmodels



# to do items:
    # GUI API
    # GUI Tutorial
    # parameters menu
        # change xtick label size
        # turn on/off xtick labels
        # verbose option, etc.
    # help menu (link to tutorial)
    # copy rows in Tables to clipboard
    # ensembles and uncertainty
    # show multiple examples of attribution maps in a given cluster
    # Page 2 option to "Load current cluster" based on cluster currently selected on page 3
    # on Page 3, 'start recording' button to visualize cluster trajectory (with relative height)
    # save/load back in a set of indices for a given cluster
    # P2 logo viewer: gray overlay over a specific site?
    # open new windows on left and right of main window (so as to not completely hide main window)
    # reopen subwindows exactly where they were on the screen
    # CLI agruments for bypassing first page
    # P2: adaptive, fixed, custom y-axis
    # P3: variants window --> Number of variants [1, ..., 10]; 
        # Position | Reference | Variant
        # [Entry]  | [Filled]  | [Dropdown]
    # P3: All clusters tab --> clusters vs. attribution errors
    # load GUI without embedding (just cluster); show N/A on embedding plots, etc.
    # marginals() --> x_skip, y_skip; c_skip missing positional skip...


################################################################################
# global assets:

py_dir = os.path.dirname(os.path.realpath(__file__)) # python file location
parent_dir = os.path.dirname(py_dir)
icon_dir = os.path.join(parent_dir, 'docs/_static/gui_icons')

progname = 'SEAM Interactive Interpretability Tool'
font_standard = QtGui.QFont('Arial', 12)

verbose = 0
dpi = 200 # or: plt.rcParams['savefig.dpi'] = dpi

################################################################################
# global widgets:

class QVLine(QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)

################################################################################
# file imports tab:

class Imports(QtWidgets.QWidget):
    mave_fname = ''
    maps_fname = ''
    manifold_fname = ''
    clusters_dir = ''
    clusters = ''
    mave_col_names = ''
    ref_full = ''
    ref_idx = 0
    hamming_orig = ''
    current_tab = 0
    cluster_col = 'Cluster'

    def __init__(self, parent=None):
        super(Imports, self).__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(20,20,20,20) # W, N, E, S
        layout.setSpacing(10)

        def load_mave():
            Imports.mave_fname = QFileDialog.getOpenFileName(self, 'Choose data file', '', ('Data files (*.csv)'))[0]
            if Imports.mave_fname:
                self.entry_mave.setDisabled(False)
                self.entry_mave.setText(Imports.mave_fname)#os.path.basename(Imports.mave_fname))
                self.entry_mave.setDisabled(True)
                Imports.mave = pd.read_csv(Imports.mave_fname)
                Imports.ref_full = Imports.mave['Sequence'][0]
                Imports.startbox_cropped.setMaximum(len(Imports.mave['Sequence'][0])-1)
                Imports.startbox_cropped.setValue(0)
                Imports.stopbox_cropped.setMaximum(len(Imports.mave['Sequence'][0]))
                Imports.stopbox_cropped.setValue(len(Imports.mave['Sequence'][0]))
                Imports.stopbox_cropped.setSuffix(' / %s' % len(Imports.mave['Sequence'][0]))
                Imports.combo_ref.setDisabled(False)
                Imports.combo_ref.setCurrentIndex(0)
                Imports.entry_ref.setReadOnly(False)
                Imports.entry_ref.setText(Imports.ref_full)
                Imports.entry_ref.setReadOnly(True)
                Imports.startbox_cropped.setEnabled(True)  
                Imports.stopbox_cropped.setEnabled(True)
            else:
                self.entry_mave.setDisabled(False)
                self.entry_mave.setText('Filename')
                self.entry_mave.setDisabled(True)
                Imports.entry_ref.setReadOnly(True)
                Imports.combo_ref.setDisabled(True)
                Imports.stopbox_cropped.setSuffix('')
                Imports.startbox_cropped.setEnabled(False)
                Imports.stopbox_cropped.setEnabled(False)
            tabs.setTabEnabled(1, False)
            tabs.setTabEnabled(2, False)
            Imports.btn_toP2.setDisabled(False)


        def select_ref():
            if Imports.combo_ref.currentText() == 'First row':
                Imports.ref_full = Imports.mave['Sequence'][0]
                Imports.entry_ref.setReadOnly(False)
                Imports.entry_ref.setText(Imports.ref_full)
                Imports.entry_ref.setReadOnly(True)
            elif Imports.combo_ref.currentText() == 'Custom':
                Imports.combo_ref.setDisabled(False)
                Imports.entry_ref.setText('')
                Imports.entry_ref.setReadOnly(False)
            elif Imports.combo_ref.currentText() == 'None':
                Imports.entry_ref.setText('')
                Imports.entry_ref.setReadOnly(True)
                Imports.ref_full = ''
            try: # can remove try/except (ZULU: needed for developer shortcut)
                tabs.setTabEnabled(1, False)
                tabs.setTabEnabled(2, False)
                Imports.btn_toP2.setDisabled(False)
            except:
                pass 


        def load_maps():
            Imports.maps_fname = QFileDialog.getOpenFileName(self, 'Choose data file', '', ('Data files (*.npy)'))[0]
            if Imports.maps_fname:
                self.entry_maps.setDisabled(False)
                self.entry_maps.setText(Imports.maps_fname)#os.path.basename(Imports.maps_fname))
                self.entry_maps.setDisabled(True)
            else:
                self.entry_maps.setDisabled(False)
                self.entry_maps.setText('Filename')
                self.entry_maps.setDisabled(True)
            tabs.setTabEnabled(1, False)
            tabs.setTabEnabled(2, False)
            Imports.btn_toP2.setDisabled(False)

        
        def cropped_signal():
            Imports.btn_toP2.setDisabled(False)


        def load_manifold():
            Imports.manifold_fname = QFileDialog.getOpenFileName(self, 'Choose data file', '', ('Data files (*.npy)'))[0]
            if Imports.manifold_fname:
                self.entry_manifold.setDisabled(False)
                self.entry_manifold.setText(Imports.manifold_fname)#os.path.basename(Imports.manifold_fname))
                self.entry_manifold.setDisabled(True)

                # bypass entering in the specific start and stop points for the cropped maps if already in filename of manifold:
                fname_contains = 'crop_'
                if Imports.manifold_fname.__contains__('_crop_'):
                    try:
                        fname_end = Imports.manifold_fname.split(fname_contains, 1)[1]
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
                        else:
                            pass
                    except:
                        pass
            else:
                self.entry_manifold.setDisabled(False)
                self.entry_manifold.setText('Filename')
                self.entry_manifold.setDisabled(True)
            tabs.setTabEnabled(1, False)
            tabs.setTabEnabled(2, False)
            Imports.btn_toP2.setDisabled(False)


        def load_clusters():
            Imports.clusters_dir = str(QFileDialog.getExistingDirectory(self, 'Select directory'))
            if Imports.clusters_dir:
                self.entry_clusters.setDisabled(False)
                self.entry_clusters.setText(Imports.clusters_dir)#os.path.basename(Imports.clusters_dir))
                self.entry_clusters.setDisabled(True)
            else:
                self.entry_clusters.setDisabled(False)
                self.entry_clusters.setText('Filename')
                self.entry_clusters.setDisabled(True)
            tabs.setTabEnabled(1, False)
            tabs.setTabEnabled(2, False)
            self.checkbox_sort.setDisabled(False)
            Imports.btn_toP2.setDisabled(False)


        def sorted_signal():
            if self.checkbox_sort.isChecked():
                Imports.cluster_col = 'Cluster_sort'
            else:
                Imports.cluster_col = 'Cluster'


        self.label_mave = QLabel('MAVE dataset')
        self.label_mave.setFont(font_standard)
        self.label_mave.setMargin(20)
        self.entry_mave = QLineEdit('Filename')
        self.entry_mave.setDisabled(True)
        self.browse_mave = QPushButton('          Browse          ', self)
        self.browse_mave.clicked.connect(load_mave)
        self.browse_mave.setToolTip('Accepted format: <i>.csv</i>')

        self.label_ref = QLabel('Reference sequence')
        self.label_ref.setFont(font_standard)
        self.label_ref.setMargin(20)
        Imports.entry_ref = QLineEdit('')
        Imports.entry_ref.setReadOnly(True)
        Imports.combo_ref = QComboBox(self)
        Imports.combo_ref.setDisabled(True)
        Imports.combo_ref.addItem('First row')
        Imports.combo_ref.addItem('Custom')
        Imports.combo_ref.addItem('None')
        Imports.combo_ref.currentTextChanged.connect(select_ref)
        Imports.combo_ref.setToolTip('Define a reference sequence for the MAVE dataset.')
        Imports.combo_ref.setFocus(True)

        self.label_maps = QLabel('Attribution maps')
        self.label_maps.setFont(font_standard)
        self.label_maps.setMargin(20)
        self.entry_maps = QLineEdit('Filename')
        self.entry_maps.setDisabled(True)
        self.browse_maps = QPushButton('          Browse          ', self)
        self.browse_maps.clicked.connect(load_maps)
        self.browse_maps.setToolTip('Accepted formats: <i>.npy</i>')

        class DoubleSpinBox(QtWidgets.QDoubleSpinBox):
            def changeEvent(self, e):
                if e.type() == QtCore.QEvent.EnabledChange:
                    self.lineEdit().setVisible(self.isEnabled())
                return super().changeEvent(e)

        self.label_start = QLabel('Map start position')
        self.label_start.setFont(font_standard)
        self.label_start.setMargin(20)
        Imports.startbox_cropped = DoubleSpinBox(self)
        Imports.startbox_cropped.setButtonSymbols(QAbstractSpinBox.NoButtons)
        Imports.startbox_cropped.setDecimals(0)
        Imports.startbox_cropped.setMinimum(0)
        Imports.startbox_cropped.setFont(font_standard)
        Imports.startbox_cropped.setDisabled(True)
        Imports.startbox_cropped.valueChanged.connect(cropped_signal)
        Imports.startbox_cropped.setToolTip('Provide the new start position with respect to MAVE sequences.')

        self.label_stop = QLabel('Map stop position')
        self.label_stop.setFont(font_standard)
        self.label_stop.setMargin(20)
        Imports.stopbox_cropped = DoubleSpinBox(self)
        Imports.stopbox_cropped.setButtonSymbols(QAbstractSpinBox.NoButtons)
        Imports.stopbox_cropped.setDecimals(0)
        Imports.stopbox_cropped.setMinimum(1)
        Imports.stopbox_cropped.setFont(font_standard)
        Imports.stopbox_cropped.setDisabled(True)
        Imports.stopbox_cropped.valueChanged.connect(cropped_signal)
        Imports.stopbox_cropped.setToolTip('Provide the new stop position with respect to MAVE sequences.')
   
        self.label_manifold = QLabel('Manifold')
        self.label_manifold.setFont(font_standard)
        self.label_manifold.setMargin(20)
        self.entry_manifold = QLineEdit('Filename')
        self.entry_manifold.setDisabled(True)
        self.browse_manifold = QPushButton('          Browse          ', self)
        self.browse_manifold.clicked.connect(load_manifold)
        self.browse_manifold.setToolTip('Accepted formats: <i>.npy</i>')

        self.label_clusters = QLabel('Clusters')
        self.label_clusters.setFont(font_standard)
        self.label_clusters.setMargin(20)
        self.entry_clusters = QLineEdit('Filename')
        self.entry_clusters.setDisabled(True)
        self.browse_clusters = QPushButton('          Browse          ', self)
        self.browse_clusters.clicked.connect(load_clusters)
        self.browse_clusters.setToolTip('Accepted formats: <i>.npy</i>')

        self.label_sort = QLabel('Clusters pre-sorted:')
        self.label_sort.setFont(font_standard)
        self.label_sort.setMargin(20)
        self.checkbox_sort = QCheckBox(self)
        self.checkbox_sort.stateChanged.connect(sorted_signal)
        self.checkbox_sort.setDisabled(True)
        self.checkbox_sort.setToolTip('Check if clusters were sorted in meta_explainer.py.')

        # end-of-tab completion:
        self.label_empty = QLabel('')
        self.line_L = QLabel('') # aesthetic line left
        self.line_L.setFont(font_standard)
        self.line_L.setMargin(0)
        self.line_L.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        self.line_R = QLabel('') # aesthetic line right
        self.line_R.setFont(font_standard)
        self.line_R.setMargin(0)
        self.line_R.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        Imports.btn_toP2 = QPushButton('Confirm Imports', self)
        Imports.btn_toP2.setToolTip('All entries must be complete.')
        
        # first row
        layout.addWidget(self.label_mave, 0, 0, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.entry_mave, 0, 1, 1, 4)
        layout.addWidget(self.browse_mave, 0, 5, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # first row, subrow
        layout.addWidget(self.label_ref, 1, 1, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(Imports.entry_ref, 1, 2, 1, 3)
        layout.addWidget(Imports.combo_ref, 1, 5, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # second row
        layout.addWidget(self.label_maps, 2, 0, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.entry_maps, 2, 1, 1, 4)
        layout.addWidget(self.browse_maps, 2, 5, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # second row, subrow
        layout.addWidget(self.label_start, 3, 1, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(Imports.startbox_cropped, 3, 2, 1, 1, QtCore.Qt.AlignLeft)
        layout.addWidget(self.label_stop, 3, 3, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(Imports.stopbox_cropped, 3, 4, 1, 1, QtCore.Qt.AlignLeft)
        # third row
        layout.addWidget(self.label_manifold, 4, 0, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.entry_manifold, 4, 1, 1, 4)
        layout.addWidget(self.browse_manifold, 4, 5, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # fourth row
        layout.addWidget(self.label_clusters, 5, 0, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.entry_clusters, 5, 1, 1, 4)
        layout.addWidget(self.browse_clusters, 5, 5, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # fourth row, subrow
        layout.addWidget(self.label_sort, 6, 1, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.checkbox_sort, 6, 2, 1, 1, QtCore.Qt.AlignLeft)
        # final section
        layout.addWidget(self.label_empty, 7, 1, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.line_L, 8, 0, 1, 2, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.line_R, 8, 4, 1, 2, QtCore.Qt.AlignVCenter)
        layout.addWidget(Imports.btn_toP2, 8, 2, 1, 2)

        self.setLayout(layout)


        if 1: # developer shortcut (ZULU)
            if 1: # local DeepSTARR locus with Ohler1
                data_dir = os.path.join(parent_dir, 'examples/examples_deepstarr/outputs_local_AP1-m0-seq13748/archives/AP1-rank0_origVersion')
                #data_dir = os.path.join(parent_dir, 'examples/examples_deepstarr/outputs_local_Ohler1-m0-seq20647')
                Imports.mave_fname = os.path.join(data_dir, 'mave.csv')
                Imports.mave_col_names = ['DNN', 'Hamming']
                Imports.combo_ref.setCurrentIndex(1)
                Imports.ref_full = pd.read_csv(Imports.mave_fname)['Sequence'].iloc[0]
                Imports.entry_ref.setText(Imports.ref_full)
                Imports.maps_fname = os.path.join(data_dir, 'maps_deepshap.npy')
                Imports.startbox_cropped.setMinimum(0)
                Imports.startbox_cropped.setMaximum(len(Imports.ref_full)-1)
                Imports.stopbox_cropped.setMinimum(1)
                Imports.stopbox_cropped.setMaximum(len(Imports.ref_full))
                if 1: # cropped attribution maps / embedding
                    Imports.startbox_cropped.setValue(110)
                    Imports.stopbox_cropped.setValue(137)
                    Imports.manifold_fname = os.path.join(data_dir, 'manifold_deepshap_umap_crop_110_137.npy')
                    if 0:
                        Imports.clusters_dir = os.path.join(data_dir, 'clusters_deepshap_umap_kmeans200_crop_110_137')
                    else:
                        Imports.clusters_dir = os.path.join(data_dir, 'clusters_deepshap_umap_kmeans200_crop_110_137_sortMedian')
                        Imports.cluster_col = 'Cluster_sort'
                else: # uncropped
                    Imports.startbox_cropped.setValue(0)
                    Imports.stopbox_cropped.setValue(len(Imports.ref_full))
                    Imports.manifold_fname = os.path.join(data_dir, 'manifold_deepshap_pca.npy')
                    Imports.clusters_dir = os.path.join(data_dir, 'clusters_pca_kmeans30')
                    #Imports.clusters_dir = os.path.join(data_dir, 'clusters_hierarchical_maxclust30_sortMedian')
            elif 0: # GLOBAL DEEPSTARR AP1-N
                data_dir = os.path.join(parent_dir, 'examples/examples_deepstarr/outputs_global_CREB-NN')
                Imports.mave_fname = os.path.join(data_dir, 'mave.csv')
                Imports.mave_col_names = ['DNN', 'GIA']
                Imports.combo_ref.setCurrentIndex(2)
                Imports.ref_full = ''
                Imports.entry_ref.setText(Imports.ref_full)
                Imports.maps_fname = os.path.join(data_dir, 'maps_deepshap.npy')
                Imports.startbox_cropped.setMinimum(0)
                Imports.startbox_cropped.setMaximum(248)
                Imports.stopbox_cropped.setMinimum(1)
                Imports.stopbox_cropped.setMaximum(249)
                if 1: # cropped attribution maps / embedding
                    Imports.manifold_fname = os.path.join(data_dir, 'manifold_deepshap_umap_crop_114_143.npy')
                    Imports.clusters_dir = os.path.join(data_dir, 'clusters_deepshap_umap_kmeans200_crop_114_143')
                    Imports.startbox_cropped.setValue(114)
                    Imports.stopbox_cropped.setValue(137)
                else: # uncropped
                    Imports.startbox_cropped.setValue(0)
                    Imports.stopbox_cropped.setValue(len(Imports.ref_full))
                    Imports.manifold_fname = os.path.join(data_dir, 'manifold_deepshap_umap.npy')
                    Imports.clusters_dir = os.path.join(data_dir, 'clusters_umap_kmeans')
            elif 0: # LOCAL CLIPNET
                data_dir = os.path.join(parent_dir, 'examples/examples_clipnet/outputs_local_PIK3R3/heterozygous_pt01/quantity_nfolds9')
                Imports.mave_fname = os.path.join(data_dir, 'mave.csv')
                Imports.mave_col_names = ['DNN', 'Hamming']
                Imports.combo_ref.setCurrentIndex(1)
                Imports.ref_full = pd.read_csv(Imports.mave_fname)['Sequence'].iloc[0]
                Imports.entry_ref.setText(Imports.ref_full)
                Imports.maps_fname = os.path.join(data_dir, 'maps_deepshap.npy')
                Imports.startbox_cropped.setMinimum(0)
                Imports.startbox_cropped.setMaximum(len(Imports.ref_full)-1)
                Imports.stopbox_cropped.setMinimum(1)
                Imports.stopbox_cropped.setMaximum(len(Imports.ref_full))
                Imports.startbox_cropped.setValue(0)
                Imports.stopbox_cropped.setValue(len(Imports.ref_full))
                Imports.manifold_fname = os.path.join(data_dir, 'manifold_deepshap_umap.npy')
                #Imports.clusters_dir = os.path.join(data_dir, 'clusters_umap_dbscan')
                #Imports.clusters_dir = os.path.join(data_dir, 'clusters_umap_kmeans200')
                Imports.clusters_dir = os.path.join(data_dir, 'clusters_hierarchical_maxclust200')
            elif 0: # TFBind
                data_dir = os.path.join(parent_dir, 'examples/examples_tfbind/outputs_8mer_Hnf4a')
                Imports.mave_fname = os.path.join(data_dir, 'mave.csv')
                Imports.mave_col_names = ['DNN']#, 'Hamming']
                Imports.combo_ref.setCurrentIndex(1)
                Imports.ref_full = pd.read_csv(Imports.mave_fname)['Sequence'].iloc[0]
                Imports.entry_ref.setText(Imports.ref_full)
                Imports.maps_fname = os.path.join(data_dir, 'maps_ism.npy')
                Imports.startbox_cropped.setMinimum(0)
                Imports.startbox_cropped.setMaximum(len(Imports.ref_full)-1)
                Imports.stopbox_cropped.setMinimum(1)
                Imports.stopbox_cropped.setMaximum(len(Imports.ref_full))
                Imports.startbox_cropped.setValue(0)
                Imports.stopbox_cropped.setValue(len(Imports.ref_full))
                Imports.manifold_fname = os.path.join(data_dir, 'manifold_ism_umap.npy')
                #Imports.clusters_dir = os.path.join(data_dir, 'clusters_umap_dbscan')
                Imports.clusters_dir = os.path.join(data_dir, 'clusters_umap_kmeans200')
                #Imports.clusters_dir = os.path.join(data_dir, 'clusters_hierarchical_maxclust200')
            elif 0: # DeepMEL2
                #data_dir = os.path.join(parent_dir, 'examples/examples_deepmel2/outputs_EFS-idx22-class16-traj20')
                #data_dir = os.path.join(parent_dir, 'examples/examples_deepmel2/outputs_EFS-idx17-class16-traj20')
                data_dir = os.path.join(parent_dir, 'examples/examples_deepmel2/outputs_REVO-idx45-class16-tw4-wl5101520-tm20-tr3')
                Imports.mave_fname = os.path.join(data_dir, 'mave.csv')
                Imports.mave_col_names = ['DNN']#, 'Hamming']
                Imports.combo_ref.setCurrentIndex(1)
                Imports.ref_full = pd.read_csv(Imports.mave_fname)['Sequence'].iloc[0]
                Imports.entry_ref.setText(Imports.ref_full)
                Imports.maps_fname = os.path.join(data_dir, 'maps_deepshap.npy')
                Imports.startbox_cropped.setMinimum(0)
                Imports.startbox_cropped.setMaximum(len(Imports.ref_full)-1)
                Imports.stopbox_cropped.setMinimum(1)
                Imports.stopbox_cropped.setMaximum(len(Imports.ref_full))
                Imports.startbox_cropped.setValue(0)
                Imports.stopbox_cropped.setValue(len(Imports.ref_full))
                Imports.manifold_fname = os.path.join(data_dir, 'manifold_deepshap_umap.npy')
                #Imports.clusters_dir = os.path.join(data_dir, 'clusters_umap_dbscan')
                #Imports.clusters_dir = os.path.join(data_dir, 'clusters_umap_kmeans200')
                Imports.clusters_dir = os.path.join(data_dir, 'clusters_hierarchical_maxclust200')
            elif 0: # ProCapNet
                data_dir = os.path.join(parent_dir, 'examples/examples_procapnet/outputs_MYC_counts_fold0_r10')
                Imports.mave_fname = os.path.join(data_dir, 'mave.csv')
                Imports.mave_col_names = ['DNN']#, 'Hamming']
                Imports.combo_ref.setCurrentIndex(1)
                Imports.ref_full = pd.read_csv(Imports.mave_fname)['Sequence'].iloc[0]
                Imports.entry_ref.setText(Imports.ref_full)
                Imports.maps_fname = os.path.join(data_dir, 'maps_deepshap.npy')
                Imports.startbox_cropped.setMinimum(0)
                Imports.startbox_cropped.setMaximum(len(Imports.ref_full)-1)
                Imports.stopbox_cropped.setMinimum(1)
                Imports.stopbox_cropped.setMaximum(len(Imports.ref_full))
                Imports.startbox_cropped.setValue(0)
                Imports.stopbox_cropped.setValue(len(Imports.ref_full))
                Imports.manifold_fname = os.path.join(data_dir, 'manifold_deepshap_umap.npy')
                Imports.clusters_dir = os.path.join(data_dir, 'clusters_hierarchical_maxclust200')

################################################################################
# custom clustering tab:    

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
    gradxinput = False
    logos_start = 0
    logos_stop = 1
    plot_ref = False

    clusters = '' #ZULU remove and set up properly

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
                if Imports.cluster_col == 'Cluster_sort': # randomize ordering of cluster colors (needed for vizualization only if clusters are sorted)
                    unique_clusters = np.unique(Predef.cluster_colors) # get unique clusters and create a randomized mapping
                    randomized_mapping = {cluster: i for i, cluster in enumerate(np.random.permutation(unique_clusters))}
                    Predef.cluster_colors = [randomized_mapping[cluster] for cluster in Predef.cluster_colors] # map cluster labels to randomized indices

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
            Custom.btn_reset.click()

        def choose_zorder():
            Custom.plt_zorder = Custom.entry_zorder.currentText()
            if Custom.entry_zorder.currentText() == 'Original':
                Custom.zord = Custom.zord0
            elif Custom.entry_zorder.currentText() == 'Ascending':
                Custom.zord = np.argsort(np.array(Imports.mave[Custom.cmap]))
            elif Custom.entry_zorder.currentText() == 'Descending':
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
            if Imports.ref_idx != '': #ZULU, add warning or disable
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
        Custom.ax.tick_params(axis='both', which='major', labelsize=10) #NEW
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
        Custom.checkbox_ref = QCheckBox(self)#'Plot reference', self)
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
        # main path and clustering buttons:
        grid_top = QGridLayout()
        grid_top.addWidget(self.toolbar, 0, 0, 1, 10)
        grid_top.addWidget(self.canvas, 1, 0, 50, 10)
        grid_top.addWidget(self.line_L, 51, 0, 1, 2, QtCore.Qt.AlignVCenter)
        grid_top.addWidget(Custom.btn_reset, 51, 2, 1, 2)
        grid_top.addWidget(self.btn_connect, 51, 4, 1, 2)
        grid_top.addWidget(self.btn_view, 51, 6, 1, 2)
        grid_top.addWidget(self.line_R, 51, 8, 1, 2, QtCore.Qt.AlignVCenter)
        # movable divider for user settings:
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

        x = Imports.manifold[:,Custom.eig1_choice]
        y = Imports.manifold[:,Custom.eig2_choice]
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

        # redraw and resize figure:
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
        #for tick in Custom.ax.xaxis.get_major_ticks():
            #tick.label.set_fontsize(4)
        #for tick in Custom.ax.yaxis.get_major_ticks():
            #tick.label.set_fontsize(4)
        Custom.ax.tick_params(axis='both', which='major', labelsize=4) #NEW
        Custom.ax.get_xaxis().set_ticks([])
        Custom.ax.get_yaxis().set_ticks([])
        Custom.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (Custom.eig1_choice+1), fontsize=6)
        Custom.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (Custom.eig2_choice+1), fontsize=6)
        Custom.ax.autoscale()
        Custom.ax.margins(x=0)
        Custom.ax.margins(y=0)
        # update colorbar:
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
            # hack to get better sensitivity of readouts from polygon:
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

            # redraw and resize figure:
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

            # update colorbar:
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
                if Custom.gradxinput is False:
                    map_avg += Imports.maps[idx,:]
                else:
                    map_avg += Imports.maps[idx,:] * squid_utils.seq2oh(seqs[idx], Imports.alphabet).flatten()
                bin_count += 1
                idxs.append(idx)
        Custom.seqs_cluster = seqs[idxs]
        map_avg /= bin_count
        #map_avg = np.reshape(map_avg, (Imports.seq_length, len(''.join(Imports.alphabet))))
        map_avg = np.reshape(map_avg, (Imports.seq_length, Imports.maps_shape[2])) # NEW
        try: # NEW
            Custom.imgAvg = squid_utils.arr2pd(map_avg, Imports.alphabet)
        except: # NEW, ZULU: may need a separate option for this to deal with two-hot encodings (e.g., CLIPNET)
            Custom.imgAvg = squid_utils.arr2pd(map_avg) # NEW
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
            # create matrix based on positional frequencies:
            seq_array_cluster = motifs.create(Custom.seqs_cluster, alphabet=Imports.alphabet)
            Custom.pfm_cluster = seq_array_cluster.counts # position frequency matrix
            pseudocounts = 0.5

            # standard PWM
            pwm_cluster = Custom.pfm_cluster.normalize(pseudocounts=pseudocounts) #https://biopython-tutorial.readthedocs.io/en/latest/notebooks/14%20-%20Sequence%20motif%20analysis%20using%20Bio.motifs.html
            Custom.seq_logo_pwm = pd.DataFrame(pwm_cluster)

            # enrichment logo (see https://logomaker.readthedocs.io/en/latest/examples.html#ars-enrichment-logo)
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
    aesthetic_lvl = 1

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

        def show_gradxinput():
            if Cluster.checkbox_gxi.isChecked():
                Custom.gradxinput = True
            else:
                Custom.gradxinput = False
            idxs = Custom.average_maps()
            self.reset()

        def color_reference():
            if Cluster.checkbox_ref.isChecked():
                Custom.logo_ref = True
            else:
                Custom.logo_ref = False
            self.reset()

        def choose_aesthetic():
            if int(Cluster.entry_aesthetic.currentText()) == 0 and Imports.alphabet != ['A','C','G','T']:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Warning)
                box.setText('<b>Input Warning</b>')
                box.setFont(font_standard)
                msg = 'Aesthetic level 0 is currently only available for the alphabet ACGT.\
                        <br /><br />\
                        Reverting to aesthetic level 1.\
                        <br />'
                box.setStandardButtons(QMessageBox.Ok)
                box.setInformativeText(msg)
                reply = box.exec_()
                Cluster.entry_aesthetic.setCurrentIndex(1)
                Cluster.aesthetic_lvl = 1
            else:
                Cluster.aesthetic_lvl = int(Cluster.entry_aesthetic.currentText())
            self.reset()

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
            cluster_stats_window = Stats()
            cluster_stats_window.setMinimumSize(10, 10)
            cluster_stats_window.show()


        self.label_view = QLabel('Logo window: ')
        self.entry_start = QDoubleSpinBox(self)
        self.entry_start.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.entry_start.setDecimals(0)
        self.entry_start.setMinimum(0)
        self.entry_start.setMaximum(Imports.seq_length-1)
        self.entry_start.setValue(Custom.logos_start)
        self.entry_start.setPrefix('Start: ')
        self.entry_start.setToolTip('Start position for viewing window.')

        self.entry_stop = QDoubleSpinBox(self)
        self.entry_stop.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.entry_stop.setDecimals(0)
        self.entry_stop.setMinimum(1)
        self.entry_stop.setMaximum(Imports.seq_length)
        self.entry_stop.setValue(Custom.logos_stop)
        self.entry_stop.setPrefix('Stop: ')
        self.entry_stop.setSuffix(' / %s' % int(Imports.seq_length))
        self.entry_stop.setToolTip('Stop position for viewing window.')

        self.btn_crop = QPushButton('Crop view')
        self.btn_crop.clicked.connect(crop_logos)
        self.btn_crop.setDisabled(False)
        self.btn_crop.setDefault(False)
        self.btn_crop.setAutoDefault(False)
        self.btn_crop.setToolTip('Visually crop diplay window for logos based on start and stop values.')

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

        if Custom.gradxinput == True:
            Cluster.checkbox_gxi.setChecked(True)
        else:
            Cluster.checkbox_gxi.setChecked(False)
        Cluster.checkbox_gxi.stateChanged.connect(show_gradxinput)
        Cluster.checkbox_gxi.setToolTip('Only show attributions for characters in the logo that appear in the reference sequence.')

        if Custom.logo_ref == True:
            Cluster.checkbox_ref.setChecked(True)
        else:
            Cluster.checkbox_ref.setChecked(False)
        Cluster.checkbox_ref.stateChanged.connect(color_reference)
        Cluster.checkbox_ref.setToolTip('Color logo according to reference sequence.')

        self.label_aesthetic = QLabel('Aesthetic level: ')
        Cluster.entry_aesthetic = QComboBox(self)
        Cluster.entry_aesthetic.addItem('0')
        Cluster.entry_aesthetic.addItem('1')
        Cluster.entry_aesthetic.addItem('2')
        Cluster.entry_aesthetic.setCurrentIndex(Cluster.aesthetic_lvl)

        Cluster.entry_aesthetic.currentTextChanged.connect(choose_aesthetic)
        Cluster.entry_aesthetic.setToolTip('Change the aesthetic appearance of the logo, with the caveat that higher aesthetic levels have higher computation times.')
        Cluster.entry_aesthetic.setFocus(True)

        layout = QGridLayout(centralwidget)
        layout.setSizeConstraint(QGridLayout.SetMinimumSize) 
        layout.addWidget(self.toolbar, 0, 0, 1, 16)
        layout.addWidget(self.canvas, 1 ,0, 50, 16)
        layout.addWidget(self.label_cluster, 51, 0, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.btn_seq_table, 51, 1, 1, 1)
        layout.addWidget(self.btn_seq_stats, 51, 2, 1, 1)
        #layout.addWidget(QVLine(), 51, 3, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.label_view, 51, 4, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.entry_start, 51, 5, 1, 1)
        layout.addWidget(self.entry_stop, 51, 6, 1, 1)
        layout.addWidget(self.btn_crop, 51, 7, 1, 1)
        #layout.addWidget(QVLine(), 51, 8, 1, 1, QtCore.Qt.AlignCenter)
        layout.addWidget(self.label_display, 51, 9, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.entry_logostyle, 51, 10, 1, 1, QtCore.Qt.AlignLeft)
        layout.addWidget(Cluster.checkbox_ref, 51, 11, 1, 1, QtCore.Qt.AlignLeft)
        layout.addWidget(Cluster.checkbox_gxi, 51, 12, 1, 1, QtCore.Qt.AlignLeft)
        layout.addWidget(self.label_aesthetic, 51, 14, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(Cluster.entry_aesthetic, 51, 15, 1, 1, QtCore.Qt.AlignLeft)
        tabs.setTabEnabled(0, False) #freezes out 1st tab
        tabs.setTabEnabled(2, False) #freezes out 3rd tab
        tabs.setTabEnabled(1, False) #freezes out parent tab
        
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        self.resize(int(self.width), int(self.height/2.)) #this changes the height
        self.setWindowTitle('Average of %s attribution maps (top) | Positional statistics of %s sequences (bottom)' % (Custom.index_enc, Custom.index_enc))
        self.show()
        #self.showMaximized()
        self.draw()


    def draw(self): # draw logos for the given cluster
        self.ax1 = self.figure.add_subplot(211)
        logo1, self.ax1 = impress.plot_logo(logo_df=Custom.imgAvg[Custom.logos_start:Custom.logos_stop], 
                                            logo_type='attribution',
                                            axis=self.ax1,
                                            aesthetic_lvl=Cluster.aesthetic_lvl,
                                            ref_seq=None,
                                            center_values=not Custom.gradxinput,
                                            )

        self.ax2 = self.figure.add_subplot(212, sharex=self.ax1)
        if Custom.logo_ref is False:
            logo2, self.ax2 = impress.plot_logo(logo_df=Cluster.seq_logo[Custom.logos_start:Custom.logos_stop], 
                                                logo_type='sequence',
                                                axis=self.ax2,
                                                aesthetic_lvl=Cluster.aesthetic_lvl,
                                                ref_seq=None,
                                                center_values=True,
                                                )
        else:
            if Imports.map_crop is True:
                ref_seq = Imports.ref_full[Imports.seq_start:Imports.seq_stop]
            else:
                ref_seq = Imports.ref_full
            logo2, self.ax2 = impress.plot_logo(logo_df=Cluster.seq_logo[Custom.logos_start:Custom.logos_stop], 
                                                logo_type='sequence',
                                                axis=self.ax2,
                                                aesthetic_lvl=Cluster.aesthetic_lvl,
                                                ref_seq=ref_seq,
                                                center_values=True,
                                                )
        self.ax2.set_xlabel('Position', fontsize=6, va='top', labelpad=6)
        self.canvas.draw()


    def reset(self):
        self.ax1.clear()
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        self.ax2.clear()
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
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
        if Imports.clusters_dir != '': # NEW, previously: is not
            tabs.setTabEnabled(2, True)
        tabs.setTabEnabled(1, True)


class SequenceTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None, *args, **kwds):
        QtWidgets.QTableWidget.__init__(self, parent)
        self.library_values = kwds['data']
        self.BuildTable(self.library_values)
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        self.resize(int(self.width), int(self.height/4.))
        
    def AddToTable(self, values):
        for k,  v in enumerate(values):
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
                elif Imports.current_tab == 2:
                    row_labels.append('Sequence %s' % Predef.seqs_cluster.index[i-2])
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
        if Imports.current_tab == 1:
            print('Sequence indices:', list(Custom.idx_encircled))
            if verbose:
                for i in Custom.seqs_cluster:
                    print(i)
        elif Imports.current_tab == 2:
            print('Consensus:', Predef.pfm_cluster.consensus)
            print('Sequence indices:', list(Predef.k_idxs))
            #if verbose:
                #for i in Predef.seqs_cluster:
                    #print(i)


class Stats(QtWidgets.QMainWindow):
    def __init__(self):
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
            self.setWindowTitle('Statistics for %s sequences in cluster %s' % (Predef.num_seqs, Predef.cluster_idx))
        self.show()

        if Imports.current_tab == 1:
            total = Custom.index_enc # total number of sequences in cluster
        elif Imports.current_tab == 2:
            total = Predef.num_seqs
        # matches of sequence positional features in cluster to reference
        self.ax1 = self.figure.add_subplot(231)
        if Imports.combo_ref.currentText() != 'None':
            ref_crop = Imports.ref_full[Imports.seq_start:Imports.seq_stop]
            ref_oh = squid_utils.seq2oh(ref_crop, Imports.alphabet)
            if Imports.current_tab == 1:
                pos_freqs = np.diagonal(np.eye(len(ref_crop))*ref_oh.dot(np.array(pd.DataFrame(Custom.pfm_cluster)).T))
            elif Imports.current_tab == 2:
                pos_freqs = np.diagonal(np.eye(len(ref_crop))*ref_oh.dot(np.array(pd.DataFrame(Predef.pfm_cluster)).T))
            self.ax1.tick_params(axis="x", labelsize=4)
            self.ax1.tick_params(axis="y", labelsize=4)
            self.ax1.bar(range(len(ref_crop)), (pos_freqs/total)*100, width=1.0)
            self.ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            self.ax1.get_xaxis().set_ticks([])
            self.ax1.get_yaxis().set_ticks([])
            self.ax1.set_xlim(0,1)
            self.ax1.set_ylim(0,1)
            self.ax1.text(.5, .5, 'N/A', horizontalalignment='center', verticalalignment='center')
        self.ax1.set_title('Cluster', fontsize=6)
        self.ax1.set_xlabel('Position', fontsize=4)
        self.ax1.set_ylabel('Reference sequence'
                            '\n'
                            r'matches ($\%$)',
                            fontsize=4)
        # matches of sequence positional features in cluster to cluster's consensus
        self.ax4 = self.figure.add_subplot(234)
        if Imports.current_tab == 1:
            consensus_seq = Custom.pfm_cluster.consensus
        elif Imports.current_tab == 2: 
            consensus_seq = Predef.pfm_cluster.consensus
        consensus_oh = squid_utils.seq2oh(consensus_seq, Imports.alphabet)
        if Imports.current_tab == 1:
            pos_freqs = np.diagonal(np.eye(len(consensus_seq))*consensus_oh.dot(np.array(pd.DataFrame(Custom.pfm_cluster)).T))
        elif Imports.current_tab == 2:
            pos_freqs = np.diagonal(np.eye(len(consensus_seq))*consensus_oh.dot(np.array(pd.DataFrame(Predef.pfm_cluster)).T))
        self.ax4.set_title('Cluster', fontsize=6)
        self.ax4.set_xlabel('Position', fontsize=4)
        self.ax4.set_ylabel('Per-cluster consensus'
                            '\n'
                            r'matches ($\%$)',
                            fontsize=4)
        self.ax4.tick_params(axis="x", labelsize=4)
        self.ax4.tick_params(axis="y", labelsize=4)
        self.ax4.bar(range(len(consensus_seq)), (pos_freqs/total)*100, width=1.0)
        self.ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        # distribution of scores in background
        self.ax3 = self.figure.add_subplot(235)
        self.ax3.set_title('Background', fontsize=6)
        self.ax3.set_xlabel('DNN scores', fontsize=4)
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
        # frequency of scores in cluster
        self.ax2 = self.figure.add_subplot(232)
        self.ax2.set_title('Cluster', fontsize=6)
        self.ax2.set_xlabel('DNN scores', fontsize=4)
        self.ax2.set_ylabel('Frequency', fontsize=4)
        self.ax2.tick_params(axis="x", labelsize=4)
        self.ax2.tick_params(axis="y", labelsize=4)
        if Imports.current_tab == 1:
            self.ax2.hist(Imports.mave['DNN'][Custom.idx_encircled], bins=100) 
        elif Imports.current_tab == 2:
            self.ax2.hist(Imports.mave['DNN'][Predef.k_idxs], bins=100)    
        self.ax2.set_xlim(self.ax3.get_xlim())
        # attribution error analysis:
        self.ax5 = self.figure.add_subplot(233)
        self.ax5.set_ylabel('Attribution errors', fontsize=4)
        self.ax5.tick_params(axis="x", labelsize=4)
        self.ax5.tick_params(axis="y", labelsize=4)

        if Imports.current_tab == 1:
            #maps_cluster = Imports.maps.reshape((Imports.maps.shape[0], Imports.seq_length, len(''.join(Imports.alphabet))))[Custom.idx_encircled]
            maps_cluster = Imports.maps.reshape((Imports.maps.shape[0], Imports.seq_length, Imports.maps_shape[2]))[Custom.idx_encircled] # NEW
            errors_cluster = np.linalg.norm(maps_cluster - np.array(Custom.imgAvg), axis=(1,2))
        elif Imports.current_tab == 2:
            #maps_cluster = Imports.maps.reshape((Imports.maps.shape[0], Imports.seq_length, len(''.join(Imports.alphabet))))[Predef.k_idxs]
            maps_cluster = Imports.maps.reshape((Imports.maps.shape[0], Imports.seq_length, Imports.maps_shape[2]))[Predef.k_idxs] # NEW
            map_avg = np.zeros(shape=(1, Imports.dim))
            bin_count = 0
            for idx in range(Imports.nS):
                if idx in Predef.k_idxs:
                    map_avg += Imports.maps[idx,:]
                    bin_count += 1
            map_avg /= bin_count
            #map_avg = np.reshape(map_avg, (Imports.seq_length, len(''.join(Imports.alphabet))))
            map_avg = np.reshape(map_avg, (Imports.seq_length, Imports.maps_shape[2])) # NEW
            try: # NEW
                Predef.imgAvg = squid_utils.arr2pd(map_avg, Imports.alphabet)
            except: # NEW, ZULU: may need a separate option for this to deal with two-hot encodings (e.g., CLIPNET)
                Predef.imgAvg = squid_utils.arr2pd(map_avg)


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
        self.ax5.set_xticks([1, 2], ['Cluster', 'Background'], fontsize=5, ha='center')
        for median in boxplot['medians']:
            median.set_color('black')
        pts_cluster = np.random.normal(1, 0.015*3, size=len(errors_cluster)) # 0.015 per 0.15 boxplot width
        self.ax5.scatter(pts_cluster, errors_cluster, s=1, c='C0', linewidth=0, zorder=-10, alpha=0.5)
        if Imports.maps_bg_on is True:
            pts_background = np.random.normal(2, 0.015*3, size=len(Imports.errors_background))
            self.ax5.scatter(pts_background, Imports.errors_background, s=1, c='C0', linewidth=0, zorder=-10, alpha=0.5)
        self.ax5.set_ylim(0, self.ax5.get_ylim()[1])
        self.canvas.draw()


class Predef(QtWidgets.QWidget):
    pixmap = ''
    coordsX = 0
    coordsY = 0
    cluster_idx = ''
    num_seqs = 0
    k_idxs = ''
    logo_dir = 'clusters_avg'
    logo_path = '/y_adaptive'
    clicked = True
    dist = 9001
    df_col = 'Entropy'
    cluster_colors = ''
    cluster_sorted_indices = ''
    clusters_idx = ''

    def __init__(self, parent=None):
        super(Predef, self).__init__(parent)

        Predef.figure = Figure(dpi=dpi)
        Predef.ax = self.figure.add_subplot(111)
        Predef.ax.set_aspect('equal')
        Predef.figure.set_tight_layout(True)
        Predef.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(Predef.canvas, self)

        def format_coord(x, y):
            #return f'x={x:1.4f}, y={y:1.4f}, cluster={Predef.cluster_idx:1.0f}, sequences={Predef.num_seqs:1.0f}'
            return f'cluster={Predef.cluster_idx:1.0f}, sequences={Predef.num_seqs:1.0f}'
        
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
            #if Predef.k_idxs == '':
            if type(Predef.k_idxs) != np.ndarray: # NEW
                cluster_warning()
            else:
                global predef_table
                seqs_cluster_plus = []
                for i in range(len(Predef.seqs_cluster)+2):
                    if i == 0:
                        seqs_cluster_plus.append(Custom.pfm_background.consensus)
                    elif i == 1:
                        seqs_cluster_plus.append(Predef.pfm_cluster.consensus)
                    elif i > 1:
                        seqs_cluster_plus.append(Predef.seqs_cluster.values[i-2])
                predef_table = SequenceTable(data=seqs_cluster_plus)
                predef_table.show()

        def open_stats_window(self):
            #if Predef.k_idxs == '':
            if type(Predef.k_idxs) != np.ndarray: # NEW
                cluster_warning()
            else:
                global predef_stats_window
                predef_stats_window = Stats()
                try:
                    predef_stats_window.close()
                except:
                    pass
                predef_stats_window.setMinimumSize(10, 10)
                predef_stats_window.show()

        def open_all_stats_window(self):
            if not isinstance(Predef.df, pd.DataFrame):            
                box = QMessageBox()
                box.setIcon(QMessageBox.Warning)
                box.setText('<b>Input Warning</b>')
                box.setFont(font_standard)
                msg = 'No CSV file(s) detected.\n\nCheck that the following file has been generated in the clusters folder:\n\ncompare_clusters.csv'
                box.setStandardButtons(QMessageBox.Ok)
                box.setInformativeText(msg)
                reply = box.exec_()
                Predef.btn_all_stats.setDisabled(True)
            else:
                global predef_all_stats_window
                predef_all_stats_window = AllStats()
                try:
                    predef_all_stats_window.close()
                except:
                    pass
                predef_all_stats_window.setMinimumSize(10, 10)
                predef_all_stats_window.show()
                Predef.btn_all_stats.setDisabled(True)

        def choose_logostyle():
            #try: ZULU: need to check if option (e.g., 'clusters_enrich') exists in folder
            if Predef.entry_logostyle.currentText() == 'Average of maps':
                Predef.logo_dir = 'clusters_avg'
                if Predef.entry_yscale.currentText() == 'Adaptive y-axis':
                    Predef.logo_path = '/y_adaptive'
                elif Predef.entry_yscale.currentText() == 'Fixed y-axis':
                    Predef.logo_path = '/y_fixed'
                Predef.entry_yscale.setDisabled(False)
            elif Predef.entry_logostyle.currentText() == 'Sequence enrichment':
                Predef.logo_dir = 'clusters_seq'
                Predef.logo_path = '/enrich'
                Predef.entry_yscale.setDisabled(True)
            elif Predef.entry_logostyle.currentText() == 'Sequence PWM':
                Predef.logo_dir = 'clusters_seq'
                Predef.logo_path = '/pwm'
                Predef.entry_yscale.setDisabled(True)
            if Predef.cluster_idx == '':
                pass
            else:
                Predef.pixmap = QtGui.QPixmap(os.path.join(Imports.clusters_dir, Predef.logo_dir+Predef.logo_path+'/cluster_%s.png' % Predef.cluster_idx))
                if Predef.pixmap.isNull():
                    print('No image found:', os.path.join(Imports.clusters_dir, Predef.logo_dir+Predef.logo_path+'/cluster_%s.png' % Predef.cluster_idx))
                Predef.label_pixmap.setPixmap(Predef.pixmap)
                Predef.pixmap.scaled(Predef.label_pixmap.width(), Predef.label_pixmap.height(), QtCore.Qt.KeepAspectRatio)
                #Predef.label_pixmap.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
                Predef.label_pixmap.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                Predef.label_pixmap.setAlignment(QtCore.Qt.AlignCenter)
            #except:
                #pass #ZULU: warning box and disable option... re-enable if gotoP2() activated again
            #self.reset()

        def choose_yscale():
            #try: ZULU: need to check if option exists in folder
            Predef.logo_dir = 'clusters_avg'
            if Predef.entry_yscale.currentText() == 'Adaptive y-axis':
                Predef.logo_path = '/y_adaptive'
            elif Predef.entry_yscale.currentText() == 'Fixed y-axis':
                Predef.logo_path = '/y_fixed'
            if Predef.cluster_idx == '':
                pass
            else:
                Predef.pixmap = QtGui.QPixmap(os.path.join(Imports.clusters_dir, Predef.logo_dir+Predef.logo_path+'/cluster_%s.png' % Predef.cluster_idx))
                if Predef.pixmap.isNull():
                    print('No image found:', os.path.join(Imports.clusters_dir, Predef.logo_dir+Predef.logo_path+'/cluster_%s.png' % Predef.cluster_idx))
                Predef.label_pixmap.setPixmap(Predef.pixmap)
                Predef.pixmap.scaled(Predef.label_pixmap.width(), Predef.label_pixmap.height(), QtCore.Qt.KeepAspectRatio)
                #Predef.label_pixmap.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
                Predef.label_pixmap.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                Predef.label_pixmap.setAlignment(QtCore.Qt.AlignCenter)
            #except:
                #pass #ZULU: warning box and disable option... re-enable if gotoP2() activated again
            #self.reset()

        def highlight_cluster():
            Predef.cluster_idx = int(Predef.choose_cluster.value())
            Predef.k_idxs =  np.array(Imports.clusters[Imports.cluster_col].loc[Imports.clusters[Imports.cluster_col] == Predef.cluster_idx].index)
            Predef.num_seqs = len(Predef.k_idxs)
            Predef.clicked = False # bypass mouse click event to force replot
            Predef.dist = 9001
            self.onclick(None)
            Predef.clicked = True

        Predef.canvas.mpl_connect('button_press_event', self.onclick)
        self.width = Predef.canvas.size().width()
        Predef.ax.format_coord = format_coord

        # canvas widgets:
        Predef.label_pixmap = QLabel()
        Predef.pixmap = QtGui.QPixmap(os.path.join(Imports.clusters_dir, Predef.logo_dir+Predef.logo_path+'cluster_0.png')) #ZULU: clusterer.py --> generate a blank png by default
        Predef.pixmap.fill(QtGui.QColor(0,0,0,0)) # initiate with blank image
        Predef.label_pixmap.setScaledContents(True)
        Predef.label_pixmap.setMaximumSize(int(self.width*3/4.), int(self.width*(3/4.)*(1.25/10.))) #ZULU: make this consistent with clusterer.py 'figsize'
        Predef.label_pixmap.setPixmap(Predef.pixmap)

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

        self.btn_seq_stats = QPushButton('Cluster Statistics')
        self.btn_seq_stats.clicked.connect(open_stats_window)
        self.btn_seq_stats.setDisabled(False)
        self.btn_seq_stats.setDefault(False)
        self.btn_seq_stats.setAutoDefault(False)
        self.btn_seq_stats.setToolTip('View statistics based on sequences in the current cluster.')

        Predef.btn_all_stats = QPushButton('Compare Clusters')
        Predef.btn_all_stats.clicked.connect(open_all_stats_window)
        Predef.btn_all_stats.setDisabled(False)
        Predef.btn_all_stats.setDefault(False)
        Predef.btn_all_stats.setAutoDefault(False)
        Predef.btn_all_stats.setToolTip('View statistics for sequences over all clusters.')

        self.label_choose_cluster = QLabel('Choose cluster: ')
        Predef.choose_cluster = QDoubleSpinBox(self)
        Predef.choose_cluster.setButtonSymbols(QAbstractSpinBox.NoButtons)
        Predef.choose_cluster.setDecimals(0)
        Predef.choose_cluster.setMinimum(0)
        Predef.choose_cluster.setFont(font_standard)
        Predef.choose_cluster.setDisabled(False)
        Predef.choose_cluster.lineEdit().returnPressed.connect(highlight_cluster) # only signal when the Return or Enter key is pressed
        Predef.choose_cluster.setToolTip('Define the index of a cluster of interest.')

        self.btn_view_cluster = QPushButton('View')
        self.btn_view_cluster.setDisabled(False)
        self.btn_view_cluster.setDefault(False)
        self.btn_view_cluster.setAutoDefault(False)
        self.btn_view_cluster.clicked.connect(highlight_cluster)
        self.btn_view_cluster.setToolTip('Navigate to a cluster in the figure using its index.')

        self.label_display = QLabel('Logo display: ')
        Predef.entry_logostyle = QComboBox(self)
        Predef.entry_logostyle.addItem('Average of maps')
        Predef.entry_logostyle.addItem('Sequence enrichment')
        Predef.entry_logostyle.addItem('Sequence PWM')
        Predef.entry_logostyle.currentTextChanged.connect(choose_logostyle)
        Predef.entry_logostyle.setToolTip('Select the logo visualization scheme.')

        Predef.entry_yscale = QComboBox(self)
        Predef.entry_yscale.addItem('Adaptive y-axis')
        Predef.entry_yscale.addItem('Fixed y-axis')
        Predef.entry_yscale.currentTextChanged.connect(choose_yscale)
        Predef.entry_yscale.setToolTip('Select the y-axis scaling used for rendering logos.')


        layout = QGridLayout()
        layout.setSizeConstraint(QGridLayout.SetMinimumSize)
        grid_top = QGridLayout()
        grid_top.addWidget(self.toolbar, 0, 0, 1, 12)
        grid_top.addWidget(Predef.canvas, 1, 0, 50, 12)
        grid_top.addWidget(Predef.label_pixmap, 51, 0, 1, 12, QtCore.Qt.AlignCenter)
        grid_top.addWidget(self.line_L, 52, 0, 1, 3, QtCore.Qt.AlignVCenter)
        grid_top.addWidget(self.btn_seq_table, 52, 3, 1, 2)
        grid_top.addWidget(self.btn_seq_stats, 52, 5, 1, 2)
        grid_top.addWidget(Predef.btn_all_stats, 52, 7, 1, 2)
        grid_top.addWidget(self.line_R, 52, 9, 1, 3, QtCore.Qt.AlignVCenter)
        grid_bot = QGridLayout()
        grid_bot.addWidget(self.label_choose_cluster, 0, 0, 1, 1, QtCore.Qt.AlignRight)
        grid_bot.addWidget(Predef.choose_cluster, 0, 1, 1, 1, QtCore.Qt.AlignCenter)
        grid_bot.addWidget(self.btn_view_cluster, 0, 2, 1, 1, QtCore.Qt.AlignLeft)
        grid_bot.addWidget(QVLine(), 0, 3, 1, 2, QtCore.Qt.AlignCenter)
        grid_bot.addWidget(self.label_display, 0, 10, 1, 1, QtCore.Qt.AlignRight)
        grid_bot.addWidget(self.entry_logostyle, 0, 11, 1, 1, QtCore.Qt.AlignCenter)
        grid_bot.addWidget(self.entry_yscale, 0, 12, 1, 1, QtCore.Qt.AlignLeft)
        #grid_bot.addWidget(QVLine(), 0, 12, 1, 2, QtCore.Qt.AlignCenter)
        
        widget_top = QWidget()
        widget_top.setLayout(grid_top)
        widget_bot = QWidget()
        widget_bot.setLayout(grid_bot)
        Predef.splitter = QSplitter(QtCore.Qt.Vertical) # movable divider
        Predef.splitter.addWidget(widget_top)
        Predef.splitter.addWidget(widget_bot)
        Predef.splitter.setStretchFactor(1,1)
        layout.addWidget(self.splitter)
        self.setLayout(layout)


    def onclick(self, event):
        splitter_sizes = Predef.splitter.sizes()
        Predef.splitter.setSizes(splitter_sizes)
        try:
            predef_table.close()
        except:
            pass
        try:
            predef_stats_window.close()
        except:
            pass
        zooming_panning = (self.figure.canvas.cursor().shape() != 0) # arrow is 0 when not zooming (2) or panning (9)
        if not zooming_panning or Predef.clicked == False:
            if Predef.clicked == True:
                ix, iy = event.xdata, event.ydata
                if ix != None and iy != None:
                    self.coordsX = float(ix)
                    self.coordsY = float(iy)
                Predef.dist, idx = spatial.KDTree(Predef.manifold).query(np.array([self.coordsX, self.coordsY]))
            if Predef.dist < .01 or Predef.clicked == False:
                # locate all members in dataframe that belong to cluster:
                if Predef.clicked == True:
                    Predef.cluster_idx = Imports.clusters[Imports.cluster_col][idx]
                    Predef.k_idxs =  np.array(Imports.clusters[Imports.cluster_col].loc[Imports.clusters[Imports.cluster_col] == Predef.cluster_idx].index)
                    Predef.num_seqs = len(Predef.k_idxs)
                    Predef.choose_cluster.setValue(Predef.cluster_idx)
                # redraw and resize figure:
                Predef.ax.clear()
                Predef.scatter = Predef.ax.scatter(Predef.pts_origX[::Custom.plt_step],
                                                    Predef.pts_origY[::Custom.plt_step],
                                                    s=Custom.plt_marker,
                                                    #c=Imports.clusters[Imports.cluster_col][::Custom.plt_step],
                                                    c=Predef.cluster_colors,
                                                    cmap='tab10', linewidth=Custom.plt_lw, zorder=0)
                Predef.scatter = Predef.ax.scatter(Predef.pts_origX[Predef.k_idxs][::Custom.plt_step],
                                                    Predef.pts_origY[Predef.k_idxs][::Custom.plt_step],
                                                    c='black', 
                                                    s=Custom.plt_marker, cmap='jet', linewidth=Custom.plt_lw, zorder=10)
                Predef.ax.tick_params(axis='both', which='major', labelsize=4) #NEW
                Predef.ax.get_xaxis().set_ticks([])
                Predef.ax.get_yaxis().set_ticks([])
                Predef.ax.set_title('Click on a cluster to view its logo', fontsize=5)
                try:
                    Predef.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % int(Imports.clusters['Psi1'][0]+1), fontsize=6)
                    Predef.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % int(Imports.clusters['Psi2'][0]+1), fontsize=6)
                except:
                    Predef.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % 0, fontsize=6)
                    Predef.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % 1, fontsize=6)
                Predef.ax.autoscale()
                Predef.ax.margins(x=0)
                Predef.ax.margins(y=0)
                # update sequence logo:
                Predef.pixmap = QtGui.QPixmap(os.path.join(Imports.clusters_dir, Predef.logo_dir+Predef.logo_path+'/cluster_%s.png' % Predef.cluster_idx))
                if Predef.pixmap.isNull():
                    print('No image found:', os.path.join(Imports.clusters_dir, Predef.logo_dir+Predef.logo_path+'/cluster_%s.png' % Predef.cluster_idx))
                Predef.label_pixmap.setPixmap(Predef.pixmap)
                Predef.pixmap.scaled(Predef.label_pixmap.width(), Predef.label_pixmap.height(), QtCore.Qt.KeepAspectRatio)
                Predef.label_pixmap.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding) #QtWidgets.QSizePolicy.Ignored
                Predef.label_pixmap.setAlignment(QtCore.Qt.AlignCenter)
                # force the matplotlib toolbar to update its status:
                if Predef.clicked == True:
                    Predef.canvas.motion_notify_event(*Predef.ax.transAxes.transform([0,0]))
                    Predef.canvas.motion_notify_event(*Predef.ax.transAxes.transform([ix,iy]))
                # save sequences in cluster
                seqs = Imports.mave['Sequence'].str.slice(Imports.seq_start, Imports.seq_stop)
                Predef.seqs_cluster = seqs[Predef.k_idxs]
                seq_array_cluster = motifs.create(Predef.seqs_cluster, alphabet=Imports.alphabet)
                Predef.pfm_cluster = seq_array_cluster.counts # position frequency matrix
                Predef.canvas.draw() # refresh canvas


class AllStats(QtWidgets.QMainWindow):
    row = 0
    col = 0
    val = 0
    sort = False#True
    threshold = 100
    delta = False
    mut_rate = 10

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
        AllStats.combo_compare.addItem('Entropy')
        AllStats.combo_compare.addItem('Entropy difference')
        AllStats.combo_compare.addItem('Reference sequence')
        AllStats.combo_compare.addItem('Consensus per cluster')
        AllStats.combo_compare.currentTextChanged.connect(self.reset)
        AllStats.combo_compare.setToolTip('Select the reference scheme for comparing the sequences in each cluster.')
        #if Imports.combo_ref.currentText() == 'None':
            #AllStats.combo_compare.setCurrentIndex(1)
            #AllStats.combo_compare.setItemData(0, False, QtCore.Qt.UserRole - 1)
        if Predef.df['Reference'].isnull().all():
            AllStats.combo_compare.setItemData(1, False, QtCore.Qt.UserRole - 1)

        AllStats.checkbox_sort = QCheckBox('Aesthetic sorting', self)
        AllStats.checkbox_sort.setChecked(AllStats.sort)
        AllStats.checkbox_sort.stateChanged.connect(self.sort_matrix)
        AllStats.checkbox_sort.setToolTip('Check to visually sort the clusters in the table based on the chosen metric.')

        self.label_mut_rate = QLabel('Mutation rate: ')
        AllStats.spin_mut_rate = QDoubleSpinBox(self)
        AllStats.spin_mut_rate.setDisabled(True)
        AllStats.spin_mut_rate.setMinimum(0.1)
        AllStats.spin_mut_rate.setMaximum(100)
        AllStats.spin_mut_rate.setDecimals(1)
        AllStats.spin_mut_rate.setSuffix('%')
        AllStats.spin_mut_rate.setValue(AllStats.mut_rate)
        AllStats.spin_mut_rate.valueChanged.connect(self.choose_mut_rate)
        AllStats.spin_mut_rate.setToolTip('Set the mutation rate used for generating the sequences in the current ensemble.')

        self.btn_marginals = QPushButton('       Marginal distributions       ')
        self.btn_marginals.setDisabled(False)
        self.btn_marginals.setDefault(False)
        self.btn_marginals.setAutoDefault(False)
        self.btn_marginals.clicked.connect(open_marginals_window)
        self.btn_marginals.setToolTip('View marginal distributions.')
        
        layout.addWidget(self.label_compare, 51, 0, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(AllStats.combo_compare, 51, 1, 1, 1, QtCore.Qt.AlignLeft)
        layout.addWidget(AllStats.checkbox_sort, 51, 2, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.label_mut_rate, 51, 3, 1, 1, QtCore.Qt.AlignRight)
        layout.addWidget(AllStats.spin_mut_rate, 51, 4, 1, 1, QtCore.Qt.AlignLeft)
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
      
        if AllStats.sort is True:
            sort = 'visual'
        else:
            if Imports.cluster_col == 'Cluster_sort':
                sort = 'predefined'
            else:
                sort = None

        AllStats.ax, AllStats.cax, AllStats.reordered_ind, AllStats.revels = impress.plot_clusters_matches_2d_gui(Predef.df, AllStats.figure, column=Predef.df_col, sort=sort, delta=AllStats.delta, mut_rate=AllStats.mut_rate, sort_index=Predef.cluster_sorted_indices)#np.array(Imports.clusters[Imports.cluster_col]))
        AllStats.canvas.mpl_connect('button_press_event', self.onclick)
        AllStats.ax.format_coord = self.format_coord
        AllStats.canvas.draw()

    def format_coord(self, x, y):
        col = int(np.floor(x))
        row = AllStats.reordered_ind[int(np.floor(y))]
        val = AllStats.revels.iloc[int(np.floor(y))][col]
        if Predef.df_col == 'Reference' or Predef.df_col == 'Consensus':
            return f'cluster={row:1.0f}, position={col:1.0f}, value={val:1.1f}%'
        elif Predef.df_col == 'Entropy':
            return f'cluster={row:1.0f}, position={col:1.0f}, value={val:1.2f} bits'
        
    def choose_mut_rate(self):
        AllStats.mut_rate = self.spin_mut_rate.value()
        self.reset()

    def sort_matrix(self):
        if AllStats.checkbox_sort.isChecked():
            AllStats.sort = True
        else:
            AllStats.sort = False
        self.reset()

    def reset(self):
        try:
            marginals_window.close()
        except:
            pass
        if AllStats.combo_compare.currentText() == 'Reference sequence':
            Predef.df_col = 'Reference'
            AllStats.delta = False
            AllStats.spin_mut_rate.setDisabled(True)
        elif AllStats.combo_compare.currentText() == 'Consensus per cluster':
            Predef.df_col = 'Consensus'
            AllStats.delta = False
            AllStats.spin_mut_rate.setDisabled(True)
        elif AllStats.combo_compare.currentText() == 'Entropy':
            Predef.df_col = 'Entropy'
            AllStats.delta = False
            AllStats.spin_mut_rate.setDisabled(True)
        elif AllStats.combo_compare.currentText() == 'Entropy difference':
            Predef.df_col = 'Entropy'
            AllStats.delta = True
            AllStats.spin_mut_rate.setDisabled(False)

        AllStats.ax.clear()
        AllStats.cax.cla()
        AllStats.figure.clear()
        if AllStats.sort is True:
            sort = 'visual'
        else:
            if Imports.cluster_col == 'Cluster_sort':
                sort = 'predefined'
            else:
                sort = None
        AllStats.ax, AllStats.cax, AllStats.reordered_ind, AllStats.revels = impress.plot_clusters_matches_2d_gui(Predef.df, AllStats.figure, column=Predef.df_col, sort=sort, delta=AllStats.delta, mut_rate=AllStats.mut_rate, sort_index=Predef.cluster_sorted_indices)#np.array(Imports.clusters[Imports.cluster_col]))
        AllStats.ax.format_coord = self.format_coord
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
                AllStats.val = f'{val:1.1f}%'
                self.open_cell_window()

    def closeEvent(self, ce): # activated when user clicks to exit via subwindow button
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
        tabs.setTabEnabled(1, True)
        tabs.setTabEnabled(2, True)
        Predef.btn_all_stats.setDisabled(False)


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
        if AllStats.combo_compare.currentText() == 'Reference sequence':
            self.setWindowTitle('Cluster: %s | Position: %s | Mismatch: %s' % (AllStats.row, AllStats.col, AllStats.val))
        else:
            self.setWindowTitle('Cluster: %s | Position: %s | Match: %s' % (AllStats.row, AllStats.col, AllStats.val))
        self.resize(int(self.width*(5/10.)), int(self.height*(5/10.)))
        self.show()

        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('Character', fontsize=4)
        self.ax.set_ylabel('Counts', fontsize=4)
        self.ax.tick_params(axis="x", labelsize=4)
        self.ax.tick_params(axis="y", labelsize=4)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        #k_idxs =  np.array(Imports.clusters[Imports.cluster_col].loc[Imports.clusters[Imports.cluster_col] == AllStats.row].index)
        for k in Predef.clusters_idx:
            k_idxs =  Imports.mave.loc[Imports.mave['Cluster'] == AllStats.row].index

        seqs = Imports.mave['Sequence'].str.slice(Imports.seq_start, Imports.seq_stop)
        seqs_cluster = seqs[k_idxs]
        occ = seqs_cluster.str.slice(AllStats.col, AllStats.col+1)
        self.vc = occ.value_counts()
        self.vc = self.vc.sort_index()
        self.vc.plot(kind='bar', ax=self.ax, rot=0)
        if AllStats.combo_compare.currentText() == 'Reference sequence':
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

        # ZULU: put this in impress.py _gui()
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

        # ZULU: put this in impress.py _gui()
        df = AllStats.revels.copy()
        self.nP = Predef.df['Position'].max()+1
        self.nC = Predef.df['Cluster'].max()+1

        # apply binary mask (counts)
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
# main window

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        #self.showMaximized()

        style = """QTabWidget::tab-bar{
            alignment: center}"""

        tab0 = Imports(self)
        tab1 = Custom(self)
        tab2 = Predef(self)

        tab0.btn_toP2.clicked.connect(self.gotoP2)

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

        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction('&About', self.about)
        fileMenu.addSeparator()
        fileMenu.addAction('&Restart', self.fileRestart)
        helpMenu = mainMenu.addMenu('&Help')
        #helpMenu.addAction('&Coordinates', self.guide_coords)

        #paraMenu = helpMenu.addMenu('&Settings')
        #paraMenu.addAction('&Parameters', self.guide_params1)
        #paraMenu.addAction('&Constraints', self.guide_params2)


    def onTabChange(self, i):
        Imports.current_tab = i
        try:
            predef_table.close()
        except:
            pass
        try:
            predef_stats_window.close()
        except:
            pass


    def gotoP2(self):
        if (Imports.mave_fname != '' and Imports.maps_fname != '' and Imports.manifold_fname != ''): # NEW, previously: is not
            Imports.btn_toP2.setDisabled(True)
            # load in silico mave and (optionally) reference seqence:
            Imports.mave = pd.read_csv(Imports.mave_fname)
            Imports.mave_col_names = list(Imports.mave)
            #if 'DNN' not in Imports.mave_col_names:
                #Custom.entry_cmap.setItemData(1, False, QtCore.Qt.UserRole - 1)
            if 'GIA' not in Imports.mave_col_names:
                Custom.entry_cmap.setItemData(1, False, QtCore.Qt.UserRole - 1)
            if 'Hamming' not in Imports.mave_col_names:
                Custom.entry_cmap.setItemData(2, False, QtCore.Qt.UserRole - 1)
            if 'Task' not in Imports.mave_col_names:
                Custom.entry_cmap.setItemData(3, False, QtCore.Qt.UserRole - 1)
            Imports.nS = len(Imports.mave)
            Imports.alphabet = sorted(list(set(Imports.mave['Sequence'][0:100].apply(list).sum())))
            Imports.seq_start = 0
            Imports.seq_stop = len(Imports.mave['Sequence'][0])
            Imports.seq_length = Imports.seq_stop - Imports.seq_start
            Imports.maps = np.load(Imports.maps_fname, mmap_mode='r') # load attribution maps (shape : (N, L, C))
            Imports.maps_shape = Imports.maps.shape # hold on to original shape for future reference
            Imports.dim = (Imports.seq_length)*Imports.maps.shape[2] #(Imports.seq_length)*len(''.join(Imports.alphabet))
            if 'Hamming' in Imports.mave_col_names:
                Imports.hamming_orig = Imports.mave['Hamming']

            Cluster.checkbox_ref = QCheckBox('Color reference', self)
            Cluster.checkbox_gxi = QCheckBox('Grad X input', self)
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
                    Imports.btn_toP2.setDisabled(False)
            else:
                Imports.ref_full = Imports.entry_ref.text()
                Cluster.checkbox_ref.setDisabled(False)
                if Imports.combo_ref.currentText() == 'None':
                    Custom.logo_ref = False
                    Cluster.checkbox_ref.setChecked(False)
                    Cluster.checkbox_ref.setDisabled(True)
                else:
                    Custom.logo_ref = True
                    Cluster.checkbox_ref.setChecked(True)
                    Cluster.checkbox_ref.setChecked(True)
                Custom.checkbox_ref.setDisabled(False)
                # reset all widgets on page 2 (if altered before change to Imports page):
                Custom.entry_cmap.setCurrentIndex(0)
                Custom.entry_theme.setCurrentIndex(0)
                Custom.entry_zorder.setCurrentIndex(0)
                Custom.checkbox_ref.setChecked(False)
                Custom.entry_stepsize.setValue(0)
                Custom.entry_markersize.setValue(.5)
                Custom.entry_eig1.setValue(1)
                Custom.entry_eig2.setValue(2)
                if int(Imports.startbox_cropped.value()) != 0 and int(Imports.stopbox_cropped.value()) != len(Imports.mave['Sequence'][0]): # if maps not pre-cropped, but need to be cropped 
                    Imports.map_crop = True
                    Imports.seq_start = int(Imports.startbox_cropped.value())
                    Imports.seq_stop = int(Imports.stopbox_cropped.value())
                    Imports.seq_length = Imports.seq_stop - Imports.seq_start
                    Imports.dim = (Imports.seq_length)*Imports.maps.shape[2] #*len(''.join(Imports.alphabet))
                    Imports.maps = Imports.maps[:,Imports.seq_start:Imports.seq_stop,:]
                    Imports.maps = Imports.maps.reshape((Imports.nS, Imports.dim))
                    # update Hamming distances based on sequence cropping:
                    ref_short = Imports.ref_full[Imports.seq_start:Imports.seq_stop]
                    if 'Hamming' in Imports.mave_col_names and Imports.combo_ref.currentText() != 'None':
                        hamming_new = []
                        for x in tqdm(Imports.mave['Sequence'], desc='Hamming update'):
                            hamming_new.append(round(spatial.distance.hamming(list(ref_short), list(x[Imports.seq_start:Imports.seq_stop])) * len(ref_short)))
                        Imports.mave['Hamming'] = hamming_new
                    # identify reference sequence (cropped) in mave dataset if present:
                    if Imports.combo_ref.currentText() == 'Custom':
                        Imports.ref_idx = ''
                        for idx, x in enumerate(Imports.mave['Sequence']):
                            if x[Imports.seq_start:Imports.seq_stop] == ref_short:
                                Imports.ref_idx = idx
                                break
                        if Imports.ref_idx == '':
                            print('No match to custom reference sequence (cropped) found in MAVE dataset.') # only affects certain plots superficially
                else:
                    Imports.map_crop = False
                    # ZULU2: if using a distal locus to infer attribution map changes, Imports.dim must be updated
                    #Imports.dim = int(768*4) # ZULU see above
                    Imports.maps = Imports.maps.reshape((Imports.nS, Imports.dim))
                    if 'Hamming' in Imports.mave_col_names:
                        Imports.mave['Hamming'] = Imports.hamming_orig
                    # identify reference sequence in mave dataset if present:
                    if Imports.combo_ref.currentText() == 'Custom':
                        Imports.ref_idx = ''
                        for idx, x in enumerate(Imports.mave['Sequence']):
                            if x == Imports.ref_full:
                                Imports.ref_idx = idx
                                break
                        if Imports.ref_idx == '':
                            print('No match to custom reference sequence found in MAVE dataset.') # only affects certain plots superficially

                if Imports.combo_ref.currentText() == 'None':
                    Imports.ref_idx = ''

                # load in manifold
                #zulu: possibly move to seam utils?
                def normalize(_d, to_sum=False, copy=True): # normalize all eigenvectors
                    d = _d if not copy else np.copy(_d) # d is an (n x dimension) array
                    d -= np.min(d, axis=0)
                    d /= (np.sum(d, axis=0) if to_sum else np.ptp(d, axis=0)) # normalize to [0,1]
                    return d
                Imports.manifold = np.load(Imports.manifold_fname)
                Imports.manifold = normalize(Imports.manifold)
                x = Imports.manifold[:,Custom.eig1_choice]
                y = Imports.manifold[:,Custom.eig2_choice]
                Custom.pts_orig = zip(x,y)
                Custom.pts_origX = x
                Custom.pts_origY = y
                Custom.scatter = Custom.ax.scatter(Custom.pts_origX[::Custom.plt_step],
                                                   Custom.pts_origY[::Custom.plt_step],
                                                   s=Custom.plt_marker,
                                                   c=Imports.mave[Custom.cmap][::Custom.plt_step],
                                                   cmap='jet', linewidth=Custom.plt_lw)
                # create initial colorbar:
                divider = make_axes_locatable(Custom.ax)
                Custom.cax = divider.append_axes('right', size='5%', pad=0.05)
                Custom.cbar = Custom.figure.colorbar(Custom.scatter, cax=Custom.cax, orientation='vertical')
                Custom.cbar.ax.set_ylabel('DNN score', rotation=270, fontsize=6, labelpad=9)
                Custom.cbar.ax.tick_params(labelsize=6)

                Custom.entry_eig1.setMaximum(Imports.manifold.shape[1])
                Custom.entry_eig2.setMaximum(Imports.manifold.shape[1])
                Custom.entry_stepsize.setMaximum(Imports.nS-1)
                # position frequency matrix of background:
                print('Calculating PFM of background...')
                seq_array_background = motifs.create(Imports.mave['Sequence'].str.slice(Imports.seq_start, Imports.seq_stop), alphabet=Imports.alphabet)
                Custom.pfm_background = seq_array_background.counts

                Custom.zord = list(range(Imports.nS))
                Custom.zord0 = Custom.zord # keep copy of original order

                Custom.logos_start = 0
                Custom.logos_stop = Imports.seq_length
                if 0:
                    print('Calculating attribution errors of background...')
                    Imports.maps_bg_on = True
                    #maps_background = Imports.maps.reshape((Imports.maps.shape[0], Imports.seq_length, len(''.join(Imports.alphabet))))
                    maps_background = Imports.maps.reshape((Imports.maps.shape[0], Imports.seq_length, Imports.maps_shape[2])) # NEW
                    Imports.errors_background = np.linalg.norm(maps_background - np.mean(maps_background, axis=0), axis=(1,2))
                else:
                    Imports.maps_bg_on = False

                tabs.setTabEnabled(2, True)
                tabs.setTabEnabled(1, True)
                tabs.setCurrentIndex(1)
                Custom.btn_reset.click()

                # predefined clusters tab:
                if Imports.clusters_dir != '':
                    Predef.cluster_idx = ''
                    Predef.ax.clear()
                    Imports.clusters = pd.read_csv(os.path.join(Imports.clusters_dir, 'all_clusters.csv'))
                    Predef.choose_cluster.setMaximum(Imports.clusters[Imports.cluster_col].max())
                    Predef.choose_cluster.setSuffix(' / %s' % Imports.clusters[Imports.cluster_col].max())
                    try:
                        x = Imports.manifold[:,int(Imports.clusters['Psi1'][0])]
                        y = Imports.manifold[:,int(Imports.clusters['Psi2'][0])]
                    except:
                        x = Imports.manifold[:,0]
                        y = Imports.manifold[:,1]
                    Predef.manifold = np.array([x, y]).T
                    Predef.pts_orig = zip(x,y)
                    Predef.pts_origX = x
                    Predef.pts_origY = y

                    Predef.clusters_idx = np.arange(Imports.clusters[Imports.cluster_col].max()+1)
                    if Imports.cluster_col == 'Cluster_sort': # needed to rearrange cells in the MSM based on sorted cluster ordering
                        boxplot_data = []
                        Imports.mave['Cluster'] = Imports.clusters['Cluster']
                        for k in Predef.clusters_idx:
                            k_idxs =  Imports.mave.loc[Imports.mave['Cluster'] == k].index
                            boxplot_data.append(Imports.mave['DNN'][k_idxs])
                        cluster_medians = [np.median(sublist) for sublist in boxplot_data] # calculate the median for each boxplot (must change this if clusters were sorted using a different metric in meta_explainer.py)
                        Predef.cluster_sorted_indices = sorted(range(len(cluster_medians)), key=lambda i: cluster_medians[i]) # get the indices of the boxplots sorted based on the median values
                    else:
                        Predef.cluster_sorted_indices = np.arange(Imports.clusters[Imports.cluster_col].max()+1)
                        Imports.mave['Cluster'] = Imports.clusters['Cluster']

                    Predef.cluster_colors = Imports.clusters[Imports.cluster_col][::Custom.plt_step]
                    if Imports.cluster_col == 'Cluster_sort': # randomize ordering of cluster colors (needed for vizualization only if clusters are sorted)
                        unique_clusters = np.unique(Predef.cluster_colors) # get unique clusters and create a randomized mapping
                        randomized_mapping = {cluster: i for i, cluster in enumerate(np.random.permutation(unique_clusters))}
                        Predef.cluster_colors = [randomized_mapping[cluster] for cluster in Predef.cluster_colors] # map cluster labels to randomized indices

                    Predef.scatter = Predef.ax.scatter(Predef.pts_origX[::Custom.plt_step],
                                                    Predef.pts_origY[::Custom.plt_step],
                                                    s=Custom.plt_marker,
                                                    #c=Imports.clusters[Imports.cluster_col][::Custom.plt_step],
                                                    c=Predef.cluster_colors,
                                                    cmap='tab10',
                                                    linewidth=Custom.plt_lw)
                    try:
                        Predef.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % int(Imports.clusters['Psi1'][0]+1), fontsize=6)
                        Predef.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % int(Imports.clusters['Psi2'][0]+1), fontsize=6)
                    except:
                        Predef.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % 0, fontsize=6)
                        Predef.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % 1, fontsize=6)

                    Predef.ax.tick_params(axis='both', which='major', labelsize=4) #NEW
                    Predef.ax.get_xaxis().set_ticks([])
                    Predef.ax.get_yaxis().set_ticks([])
                    Predef.ax.set_title('Click on a cluster to view its logo', fontsize=5)
                    Predef.ax.autoscale()
                    Predef.ax.margins(x=0)
                    Predef.ax.margins(y=0)
                    Predef.canvas.draw() # refresh canvas
                    # initiate logo canvas with blank image
                    Predef.pixmap = QtGui.QPixmap(os.path.join(Imports.clusters_dir, Predef.logo_dir+Predef.logo_path+'cluster_0.png')) #ZULU: clusterer.py --> generate a blank png by default
                    Predef.pixmap.fill(QtGui.QColor(0,0,0,0))
                    Predef.label_pixmap.setPixmap(Predef.pixmap)
                    Predef.label_pixmap.setAlignment(QtCore.Qt.AlignCenter)
                    Predef.k_idxs = ''
                    # check for files needed for subwindows on page 3:
                    if os.path.isfile(os.path.join(Imports.clusters_dir, 'compare_clusters.csv')):
                        Predef.df = pd.read_csv(os.path.join(Imports.clusters_dir, 'compare_clusters.csv'))
                    else:
                        Predef.df = ''

                    # reset widgets on page 3
                    Predef.choose_cluster.setValue(0)
                    Predef.entry_logostyle.setCurrentIndex(0)
                    Predef.entry_yscale.setCurrentIndex(0)
                    Predef.btn_all_stats.setDisabled(False)
                else:
                    tabs.setTabEnabled(2, False)

                print('Loading complete.')

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
                                %s Evan E Seitz, David M McCandlish, Justin B Kinney, Peter K Koo\
                                <br />CSHL, 2024\
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
                                </span>' % (u"\u00A9"))
        box.setStandardButtons(QMessageBox.Ok)        
        ret = box.exec_()



if __name__ == '__main__':
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    QtCore.QCoreApplication.setApplicationName(progname)
    QApplication.setStyle(QStyleFactory.create('Fusion'))
        
    # set app icon for tray:
    app_icon = QtGui.QIcon()
    app_icon.addFile(os.path.join(icon_dir, '256x256.png'), QtCore.QSize(256,256))
    app.setWindowIcon(app_icon)
        
    w = MainWindow()
    w.setWindowTitle('%s' % progname)
    w.show()
    sys.exit(app.exec_())