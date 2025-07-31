"""
# SEAM example of local library using ChromBPNet to analyze PPIF foreground and background signals at human enhancers and promoters

Model:
    - ChromBPNet
    - Model requirements: https://github.com/kundajelab/chrombpnet

Parameters:
    - 100,000 sequences
    - 10% mutation rate
    - Integrated gradients
    - Hierarchical clustering (ward)

Tested on GPUs using:
    - Python 3.8.5
    - TensorFlow 2.8.0
    - TensorFlow Probability 0.15.0
    - Keras 2.8.0
    - NumPy 1.23.4

Computation time:
    - NVIDIA GeForce RTX 2080 Ti, CUDA 11.4):
        - Saliency maps (GPU, batch_size=32): 750 seconds (12.5 minutes)
        - Clustering:
            - Distance matrix (GPU): 270 seconds (4.5 minutes)
            - Linkage (GPU): 770 seconds (12.8 minutes)

Note on multiple folds:
    This script processes one ChromBPNet fold at a time. To analyze all folds:
    1. Run with fold_index = 0, save results
    2. Continue for folds 1-4
    3. Combine results across folds for final analysis
    Each run will save attribution maps with the fold index in the filename.

Note on model type:
    If model encounters a mashable error, see FAQs on ChromBPNet GitHub page for conversion to compatible format:
    https://github.com/kundajelab/chrombpnet/blob/master/chrombpnet/helpers/postprocessing/reformat_chrombpnet_h5.py
    e.g., python reformat_chrombpnet_h5.py -cnb model.chrombpnet_nobias.fold_0.ENCSR000EOT.h5 -bm model.bias_scaled.fold_0.ENCSR000EOT.h5 -o .
"""

import os, sys, time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import squid
import pandas as pd
from tqdm import tqdm
from urllib.request import urlretrieve

np.random.seed(42)
random.seed(42)

if 0: # Use this for local install (must 'pip uninstall seam' first)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # TODO: turn this off
from seam import Compiler, Attributer, Clusterer, MetaExplainer, Identifier
from seam.logomaker_batch.batch_logo import BatchLogo

# =============================================================================
# Key hyperparameters
# =============================================================================
model_type = 'dnase_thp1' # {'dnase_thp1', 'dnase_thp1_nobias', 'dnase_k562', 'dnase_jurkat', 'dnase_jurkat_stim', 'atac_k562'}
fold_index = 0  # Choose which fold to use (0-4)
task_type = 'counts'  # {'profile', 'counts'} for logits_profile (0), logcount (1)
enhancer_or_promoter = 'promoter'  # {'promoter', 'enhancer'} of PPIF gene
mut_rate = 0.1  # mutation rate for in silico mutagenesis
num_seqs = 100000  # number of sequences to generate
n_clusters = 200 # number of clusters for hierarchical clustering
attribution_method = 'intgrad'  # {saliency, smoothgrad, intgrad, ism, deepshap}

# =============================================================================
# Overhead user settings
# =============================================================================
try:
    gpu = len(tf.config.list_physical_devices('GPU')) > 0  # Whether to use GPU (Boolean)
except:
    gpu = len(tf.config.experimental.list_physical_devices('GPU')) > 0
save_figs = True  # Whether to save quantitative figures (Boolean)
render_logos = True  # Whether to render sequence logos (Boolean)
save_logos = True  # Whether to save sequence logos (Boolean); render_logos must be True
dpi = 200  # DPI for saved figures
save_data = True  # Whether to save output data (Boolean)
delete_downloads = False  # Whether to delete downloaded models/data after use (Boolean)

# If starting from scratch, set all to False:
load_previous_library = False  # Whether to load previously-generated x_mut and y_mut (Boolean)
load_previous_attributions = False  # Whether to load previously-generated attribution maps (Boolean)
load_previous_linkage = False  # Whether to load previously-generated linkage matrix (Boolean)

# =============================================================================
# Initial setup based on user settings
# =============================================================================
# Create assets_chrombpnet folder if it doesn't exist
py_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(py_dir, 'assets_chrombpnet')
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)

if save_data:
    save_path = os.path.join(py_dir, f'outputs_chrombpnet_local_ppif_{enhancer_or_promoter}_fold{fold_index}_{attribution_method}_{model_type}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save_figs:
        save_path_figs = save_path
    else:
        save_path_figs = None
else:
    save_path = None

if load_previous_library is True: # save_data must be True
    x_mut = np.load(os.path.join(save_path, 'x_mut.npy'))
    y_mut = np.load(os.path.join(save_path, 'y_mut.npy'))
    mave_df = pd.read_csv(os.path.join(save_path, 'mave_df.csv'))

if load_previous_attributions: # save_data must be True
    attributions = np.load(os.path.join(save_path, f'attributions_{attribution_method}.npy'))

if load_previous_linkage: # save_data must be True
    link_method = 'ward'
    linkage = np.load(os.path.join(save_path, f'hierarchical_linkage_{link_method}_{attribution_method}.npy'))
    clusterer = Clusterer(
        attributions,
        gpu=gpu
    )

if task_type == 'profile':
    task_index = 0
elif task_type == 'counts':
    task_index = 1

# =============================================================================
# Download and import ChromBPNet model and data and define helper functions
# =============================================================================
def download_if_not_exists(url, filename):
    """Download a file if it doesn't exist locally."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urlretrieve(url, filename)
    else:
        print(f"Using existing {filename}")

def softmax(x, temp=1):
    """Apply softmax function to input."""
    norm_x = x - np.mean(x,axis=1, keepdims=True)
    return np.exp(temp*norm_x)/np.sum(np.exp(temp*norm_x), axis=1, keepdims=True)

def predict_tracks(model, sequence_one_hot):
    """Predict tracks for a single sequence."""
    profile_probs_predictions = []
    counts_sum_predictions = []
    preds = model.predict_on_batch(sequence_one_hot[None, ...])
    profile_probs_predictions.extend(softmax(preds[0]))
    counts_sum_predictions.extend(preds[1][:,0])
    # Return first profile prediction and counts
    return profile_probs_predictions[0], counts_sum_predictions[0]

class ChromBPNetPredictor(squid.predictor.BasePredictor):
    def __init__(self, pred_fun, task_idx=0, batch_size=64, compress_fun=None, save_dir=None, save_window=None, **kwargs):
        self.pred_fun = pred_fun
        self.task_idx = task_idx
        self.batch_size = batch_size
        # Default compression functions for each task type
        self.compress_fun = compress_fun or {
            0: Attributer.bpnet_profile,  # Profile task (index 0)
            1: lambda x: x  # Counts task (index 1)
        }[task_idx]
        squid.predictor.BasePredictor.save_dir = save_dir
        self.kwargs = kwargs

    def __call__(self, x, x_ref, save_window):
        """Required by SQUID API to make predictions."""
        return self.predict_in_batches(x, x_ref, self.pred_fun, batch_size=self.batch_size, task_idx=self.task_idx, save_window=save_window)

    def predict_in_batches(self, x, x_ref, model_pred_fun, batch_size=None, task_idx=None, save_window=None, **kwargs):
        if save_window is not None:
            x_ref = x_ref[np.newaxis,:].astype('uint8')
        N, L, A = x.shape
        num_batches = int(np.floor(N/batch_size))
        pred = []
        
        # Use instance task_idx if not provided
        task_idx = task_idx or self.task_idx
        
        for i in tqdm(range(num_batches), desc="Inference"):
            x_batch = x[i*batch_size:(i+1)*batch_size]
            if save_window is not None:
                x_ref_start = np.broadcast_to(x_ref[:,:save_window[0],:], (x_batch.shape[0],save_window[0],x_ref.shape[2]))
                x_ref_stop = np.broadcast_to(x_ref[:,save_window[1]:,:], (x_batch.shape[0],x_ref.shape[1]-save_window[1],x_ref.shape[2]))
                x_batch = np.concatenate([x_ref_start, x_batch, x_ref_stop], axis=1)
            
            pred_ix = model_pred_fun(x_batch.astype(float))
            
            # Select task output and apply compression
            task_output = pred_ix[task_idx]
            compressed = self.compress_fun(task_output)
            pred.append(compressed)
        
        # Handle remaining sequences
        if num_batches*batch_size < N:
            x_batch = x[num_batches*batch_size:]
            if save_window is not None:
                x_ref_start = np.broadcast_to(x_ref[:,:save_window[0],:], (x_batch.shape[0],save_window[0],x_ref.shape[2]))
                x_ref_stop = np.broadcast_to(x_ref[:,save_window[1]:,:], (x_batch.shape[0],x_ref.shape[1]-save_window[1],x_ref.shape[2]))
                x_batch = np.concatenate([x_ref_start, x_batch, x_ref_stop], axis=1)
            
            pred_ix = model_pred_fun(x_batch.astype(float))
            
            # Select task output and apply compression
            task_output = pred_ix[task_idx]
            compressed = self.compress_fun(task_output)
            pred.append(compressed)
        
        preds = np.concatenate(pred, axis=0)
        return preds.reshape(-1, 1)  # Ensure 2D output for SQUID

# Download losses script
losses_url = "https://raw.githubusercontent.com/kundajelab/chrombpnet/master/chrombpnet/training/utils/losses.py"
losses_path = os.path.join(assets_dir, "losses.py")
download_if_not_exists(losses_url, losses_path)

# Import losses
sys.path.append(assets_dir)
import losses

# TODO: below, get models from Zenodo
# Download model file for selected fold
if model_type == 'dnase_thp1':
    model_url = f"https://drive.google.com/uc?id=126xlwtTBBUt5JbEorzQaKaQdV9echJ4w"  # fold 0
    if fold_index == 1:
        model_url = "https://drive.google.com/uc?id=12lkf1Fj0t0ThuXdKtVtusl6oBQNi3hA9"
    elif fold_index == 2:
        model_url = "https://drive.google.com/uc?id=1w6jDVdsUcDFBiTA4IBiN1LqVEGdPYw2C"
    elif fold_index == 3:
        model_url = "https://drive.google.com/uc?id=1xt3aV06Gz5WFp4oTphikr2tvAan1Op2L"
    elif fold_index == 4:
        model_url = "https://drive.google.com/uc?id=1cmMB4wY9tcWtc-t3IN2LHDvxYgp1A_Ik"
    model_path = os.path.join(assets_dir, f"dnase_thp1_fold{fold_index}_nobias.h5")

elif model_type == 'dnase_thp1_composite':
    model_url = f"https://drive.google.com/uc?id=1X6MWSuSPE4qX4uD_3HOMcBnAQAX2yvKi"  # fold 0
    if fold_index > 0:
        print("DNase-seq THP1_nobias model fold %s not implemented yet" % fold_index)
    model_path = os.path.join(assets_dir, f"dnase_thp1_fold{fold_index}_composite.h5")

elif model_type == 'dnase_k562':
    model_url = "https://drive.google.com/uc?id=1eiulVO-7_jZo0koB-Bg_a-_MGBIPwPBe" # fold 0
    if fold_index > 0:
        print("DNase K562 model fold %s not implemented yet" % fold_index)
    model_path = os.path.join(assets_dir, f"dnase_k562_fold{fold_index}_nobias.h5")

elif model_type == 'dnase_jurkat':
    model_url = "https://drive.google.com/uc?id=1c_Zx1MrIhTwSkpo3pChK2Nkdj2tWE0fl" # fold 0
    if fold_index > 0:
        print("DNase-seq jurkat model fold %s not implemented" % fold_index)
    model_path = os.path.join(assets_dir, f"dnase_jurkat_fold{fold_index}_nobias.h5")

elif model_type == 'dnase_jurkat_stim':
    model_url = "https://drive.google.com/uc?id=1uIyWItsuGxhRGRMZhKKEMJbXypVEbAtU" # fold 0
    if fold_index > 0:
        print("DNase-seq jurkat_stim model fold %s not implemented" % fold_index)
    model_path = os.path.join(assets_dir, f"dnase_jurkat_stim_fold{fold_index}_nobias.h5")

elif model_type == 'atac_k562':
    model_url = "https://drive.google.com/uc?id=1D8mvnZDB6fFgOyjI158O6H-RAs80LR14" # fold 0
    if fold_index > 0:
        print("ATAC K562 model fold %s not implemented yet" % fold_index)
    model_path = os.path.join(assets_dir, f"atac_k562_fold{fold_index}_nobias.h5")

download_if_not_exists(model_url, model_path)

# Load model
custom_objects = {
    "multinomial_nll": losses.multinomial_nll,
    "tf": tf,
}
tf.keras.utils.get_custom_objects().update(custom_objects)
model = tf.keras.models.load_model(model_path)

# =============================================================================
# Sequence retrieval and processing
# =============================================================================
if enhancer_or_promoter == 'promoter':
    tss_pos = 81107224 # ppif tss (hg19)
    bin_number = 3 # chosen to match enformer settings (i.e., bin_size=128)
    map_crop = [190, 690] # crop attribution maps to focus on region of PPIF promoter
elif enhancer_or_promoter == 'enhancer':
    tss_pos = 81046461
    bin_number = 8 # chosen to match enformer settings
    map_crop = [730, 1230] # crop attribution maps to focus on region of PPIF enhancer

alphabet = ['A','C','G','T']
seq_length = int(model.input_shape[1]) # 2114
start = tss_pos - seq_length // 2
end = tss_pos + seq_length // 2
map_start = (seq_length // 2) - (bin_number*128)
map_end = (seq_length // 2) + (bin_number*128)

if 1:
    if enhancer_or_promoter == 'promoter':
        seq = 'TTATCCTAAGAACAGACACGAGAAAAAAGCAGGATGAGGAGCATGATCGCACCCTTACCCTCCAGATGGGAAACTGAGGCCCAGAGAGGCCTGGAGCACTGGTGCAAGCTGGTAGCAATACTCCAGGGCTCCCTAGCCCTAGCCAGTGCCTTCCTGTGTCCAGGCCTCAGCTTCCCCTGGGGACAGTACAAGTGTTGGGCCAGATGCTCTCTCAGGTCCCTGCGTGGCATCCATTTATTGCAGACCTCCCACCACCTTGCAGTGAGGGTGGCTCTGGCTGCTGGGAAGCCCACTTTCAGCACGTGGGGTTTAAATGCTCCTGCTGTGGGTCTCCCCACTGAGTCCCCTCCCAGGGTGCATGCTGATGGGAGGGGGCAGCTGGCAGTCTGCCCCGGGGCTGTCAGTGTGGGTCCTAGGAGGAGGGATCAGACCCAGTCCTGGGGAGGGCTGGGGGCCTGAAAGGAGCATGATGAGCCCAGGCTGCGTTTTCAGTCTTGCTAGAAGCAGGTCTGGTCCCCAGAAACAACTAGAGACAGGCCCAGGCCGGGACAAGCAGGCTAGGGGGCATCATGGGAGGTGTCTCAGCTTATCTCCTCCCTCTTGCGCCTCTAGCTCACTAATCCCGCCTCTCATCTCACCTTTCTCTAACCCTCTCAGACTGCAGGACCTAGGGCAGCAGGGAAGCTTATTTGGCCTGAGCCTTACCTGCAAAGGGCTCAAAGGTAGACTTTTGCTATTATATTCAAATTGCAACGTATAGATGCTCATATTTTGGAATTAATTACGTTTTAGCAGTTTTTCTTTTTTTAAAAAAATCTGCTGGTACCTTAATAAACATGGAACCAGCAACCTTTTGCCAGGTGTCTGTTCTCCGGGTCTCTGGGCCTGGAGGCGGGAGTGTTTCAAAGCACTTCACGCTCCGCGGCCCCACCGGCTGGCTGCGCTGCCCGCTGCGGCCGGCAGGGGTAGTCCACGGACAGGCCTGGAGGAGGCGGGACGGGGGCAGGGCCGGGAACCTGGGCAAGCCAATAAAGGCTGCGGCGCGCGGCTGCGCGGGACTCGGCCTTCTGGGCGCGCGCGACGTCAGTTTGAGTTCTGTGTTCTCCCCGCCCGTGTCCCGCCCGACCCGCGCCCGCGATGCTGGCGCTGCGCTGCGGCTCCCGCTGGCTCGGCCTGCTCTCCGTCCCGCGCTCCGTGCCGCTGCGCCTCCCCGCGGCCCGCGCCTGCAGCAAGGGCTCCGGCGACCCGTCCTCTTCCTCCTCCTCCGGGAACCCGCTCGTGTACCTGGACGTGGACGCCAACGGGAAGCCGCTCGGCCGCGTGGTGCTGGAGGTGAGACCGCTCGCAGGGCCGGCCTGGGCGCGGGACACGGGCCCGGGGAGAGCCCTGGGCCCCGGGCGGCGCGGTGCCGGGCGCGCTGGGTGACCTTGGGCCTCCCCATGCCGAGCTCTGGGCCTCAGTTTCCCCATTTCTGAGAATGGGCGTCAGAATGATTTCTTCCGGCCTCCCTCGGAGCACTGGAGCGGGGGAGACGGGAGGGAGGGCACGTGTGGAGGAGAAAGCTCAAGGTCAGATCGCAGAGAGGGAGGGCTCAGCACCTCTGGGCCGGCCCGGGCACGAGGGAGGGGCTCCAGGAGCCTTCTGGGGCTGAGCCTAGATCCGGAGCTCCGAGGTGGGTGTCGGGGGTCTTGGGGTGAGCGTCGTGGCCCAGCGGGTGCTCACGTGGCGGCCCTTGCACAACACGGAGCGCTTCCTGGCTCCGGCCCCGCCCCTGCGGTCGGGCTCACACTGGGGGTGCTGGGAAATGGAGCGAGAGGTGGTTTCCAGCAGTAGTGCGGGCCCAGTAGGCCTCAGGCCCCGGCCACCTGGTGGACCCCAGAATGCCCCTCCTGCGAGTCGGGACACACTCAGAGACAGTGTGCCCGGCGCCTCAACCCCTGCCACTGTCCTTGGGGGCCACACTGAGCACCTCCCCTAACTCTGTTTTTGGGTCTTTTCTAAAGCAAAGTAAGAAACAGTCACCAGGGTAGCTTTAGAGGGAAAGCCCTAGTGGAGCCTTCAGGTCGGCCACACATTGACAGCAGGGGTCTGTTTGTGTTGACTGGCCCGTATCC'
    elif enhancer_or_promoter == 'enhancer':
        seq = 'CTGGCACATTTTTTTGTTTAGCAAAGGTTTCCTGAGCCCTATTGGTGTGTGAAGACATCATGCTTGGCTCAGGGAGCCCAAGAAATCAGTGACAGCTCCTGAGGCAGAAGGGAGAAGGCAGAGGGTTCTCACTCTTGCCAGGAATGCGAAGAGCCAGTGGAGTTCAGTACCTTCTGCAAGGGCTTCCCGGAGGAGGTGGCCTTTGGAAGAGCCTGGAAAGGATGAATGGCATTTTCAAGGTGGGATGGAACTTGCCAGCCTGAGAAGAGGCACAGAGGCCCCTGTATTAGGGAGCAGTTGGCAAATGGAACCGTTGCTGCTGACATTGCAGATGGCTTCTGGCCTTGGTGCTTCTGTGGGTGGAGCAGGATTGTTGGGGCCGTTTTGGCTGGACAGGACTGTGGCTGGGAGCAGCTTATGGTGTGATACTGGGCAGGCCATGCTGGATTAGGGTTGTACAGAGTGGGACAGAATTATTAAAAAATGCCAAGCACCTGTGACCAGGTGAAGCAAGTCTGCAGGTTAAACTGTGCTCGTCGGAGAGAAGAAGGGCTCTCAGGCACCCTCCAGACCCTCCTCCCATCCCAGCCAGGCCATGCCTTTCTGGTAGCAGCCCAGGGGTCTGCTTGGTCCTTCATCCTTGGGAACAGGGACCAGGCATTCCCCCGGTGGTGGTATCTTGAGTCTAGGCAGTGGCGTCACATCAGCAGGGCTGCTGTGGGGGCCTGCGGGGGCGAGAGCCTTCTCTGCGACAGCAGGAGCCCGTCTCCCCCAGAATGAGATGAGTCACGGTCCTGTCGGTGCGGCTGCCACTTCTGCTTCTGCAGCAGGTGCGTGTGGAGGCAAGCTAGTCCTGAGACTGCCCCACACGCTGTGCCTGTCTCTTCCCCAGTGCACGGGCAGCATGCCAGCAGGGCCTCGCGTGTGTGCACGGATGTGTCCTCCATATGCCTGCGTGCATCTTTGCTCTGGGGCGGCAAACCTTCCAGCCAGGCAGTGGGATGGCCAGTTTGGGAACGTTGGTTAGCTTTGCTTCTCACCAAGTTTGGGGCAGAGGGGACAAAACAGGAAGGAGCTGGTTTCCGCCCACAAGGCTCACCCACGGAGCTGCAGGGCAGTTGGGAGCTTGTTCGTGGGCCTGTGTAGCGTCCTGGGCTGAGGTGATGAGGCAGCAGTTGCCCACTCCCTGGCCACAGGCAGGGCCCTGCCGGCCCCTGGGGCCTCCCATGGGGGTTCATTTCCTGAGCCTGGCCAGGCATGTTCTCAAAGGATAGGAACTTCCGTCTCAAGGCCATCAGCCTGGAGGGTTGGTGGGAACTTGTGGGGCAGAAGTTCTGGGGCGAGAAGCCACTCAGGCTTGGGTTTTGTTCAGATGTAGGGGACCCCCTCTGGTCATCCAGCACACCTAGGACATGGGCCTTGAGGACAGCAGGTGTCCAGGGGATCTCCTCTTTGTTTTGTAAAAGCGTGACCTGTAGAGCCTCCCTGTGAGACTCTGGTGTGCCAGTCTGGAACCAGGTGGTGTGCCGGGTCCCCAGTGACCTCCCTGGGAGACAGAGGCTGTTCCAGCCCAGCCTTCCTGAGGAGCAGGCTCATGGCTACTTTCCTGTGTTTCTCTGCACTGCCCACTCCCTCTGGGCTCCCAGTCTCCAATCCGCCACCTGGAGAGCAGAGTCTGTCGCTATCTGTCACCTTTGGCCCAAGCATCTGGATCTTCAACAAGCCACTACCCATCATTGTGCCTGGCCAACTGAGTGCCCTGAGTACAGCTCCTCTACTCCAAATGCGAGTCCTGCAGTTTCCCCAGAAATCACGAGCAGCGCTCAGTTGGGAAAGCCCTTTGACTGCATTCTCCTCTTTGAGCCACATCAGATTTAGGCTGCTGAACCTGTGTTCAAACCTATTTGACCAAGATCCACGATAAATACACATACGACACCTTGACCCAAGGTGCACATGTATTTGTGTATAAAACCAAGACAGAAGCTCAAGGACTAATCTTCATCTTTATTGTGTGCTCACACTCCATCACTTTCTAGTTGATGCTTTTTTCTCCCCTGCTGGTTGGATGTCATCACCTGCTCAGATTCTGTAACCCACAGTTGGAAAACACA'
else: # download fasta file (estimated time: 1 minute)
    import pysam # 'pip install pysam', for fasta file
    import gzip
    fasta_file = os.path.join(assets_dir, 'hg19.fa')
    if not os.path.exists(fasta_file) or os.path.getsize(fasta_file) == 0:
        if os.path.exists(fasta_file):
            os.remove(fasta_file)
        print("Downloading hg19.fa...")
        url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
        try:
            import requests
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with gzip.GzipFile(fileobj=response.raw) as gz:
                with open(fasta_file, 'wb') as f:
                    f.write(gz.read())
        except Exception as e:
            print(f"Error downloading FASTA file: {e}")
            raise
    fasta_open = pysam.Fastafile(fasta_file)
    chrom = 'chr10'
    seq = fasta_open.fetch(chrom, start, end).upper()

# convert to one-hot:
x_ref = squid.utils.seq2oh(seq, alphabet)
x_ref = np.expand_dims(x_ref,0)

# =============================================================================
# View model outputs for reference sequence
# =============================================================================
y_profiles, y_counts = predict_tracks(model, x_ref[0])

print('Wild-type counts:', y_counts)
fig, ax = plt.subplots(figsize=(20,1.5))
ax.axhline(0, color='k', linewidth=.5)
ax.bar(range(y_profiles.shape[0]), y_profiles, width=-2, color='r')
plt.title('Wild-type counts: %s' % y_counts)
plt.tight_layout()
if save_figs:
    fig.savefig(os.path.join(save_path_figs, 'wildtype_profile.png'), facecolor='w', dpi=dpi, bbox_inches='tight')
    plt.close()
else:
    plt.show()

# =============================================================================
# SQUID API
# Create in silico mutagenesis library
# =============================================================================
mut_window = [map_start, map_end]

if load_previous_library is False:
    t1 = time.time()

    # Set up predictor class for in silico MAVE
    pred_generator = ChromBPNetPredictor(
        pred_fun=model.predict_on_batch,
        task_idx=task_index,
        batch_size=512
    )

    # Set up mutagenizer class for in silico MAVE
    mut_generator = squid.mutagenizer.RandomMutagenesis(
        mut_rate=mut_rate,
        seed=42
    )

    # Generate in silico MAVE
    mave = squid.mave.InSilicoMAVE(
        mut_generator,
        pred_generator,
        seq_length,
        mut_window=mut_window
    )

    x_mut, y_mut = mave.generate(x_ref[0], num_sim=num_seqs)

    t2 = time.time() - t1
    print('Inference time:', t2)

    if save_data:
        np.save(os.path.join(save_path, 'x_mut.npy'), x_mut)
        np.save(os.path.join(save_path, 'y_mut.npy'), y_mut)

# plot histogram of deepnet predictions
fig = squid.impress.plot_y_hist(
    y_mut,
    save_dir=save_path_figs
)
if not save_path_figs:  
    plt.show()

# =============================================================================
# SEAM API
# Compile sequence analysis data into a standardized format
# =============================================================================
if load_previous_library is False:
    # Initialize compiler
    compiler = Compiler(
        x=x_mut,
        y=y_mut,
        x_ref=x_ref,
        y_bg=None,
        alphabet=alphabet,
        gpu=gpu
    )

    mave_df = compiler.compile()
    if save_data:
        mave_df.to_csv(os.path.join(save_path, 'mave_df.csv'), index=False)

ref_index = 0 # index of reference sequence (zero by default)
print(mave_df)

# =============================================================================
# SEAM API
# Compute attribution maps for each sequence in library
# =============================================================================
if load_previous_attributions is False:
    if attribution_method == 'deepshap':
        try:
            # Disable eager execution first
            tf.compat.v1.disable_eager_execution()
            tf.compat.v1.disable_v2_behavior()
            print("TensorFlow eager execution disabled for DeepSHAP compatibility")
            
            # Import SHAP to configure handlers
            import shap
            # Handle AddV2 operation (element-wise addition) as a linear operation
            shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough

            # Load the model after eager execution is disabled
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
            # Rebuild model to ensure proper graph construction
            _ = model(tf.keras.Input(shape=model.input_shape[1:]))
            
        except Exception as e:
            print(f"Warning: Could not setup TensorFlow for DeepSHAP. Error: {e}")
            print("DeepSHAP may not work properly.")
        
        # Determine compression function based on task type
        if task_type == 'profile':
            compress_fun = Attributer.bpnet_profile
        elif task_type == 'counts':
            compress_fun = Attributer.bpnet_counts
        
        # Create attributer for DeepSHAP
        attributer = Attributer(
            model,
            method=attribution_method,
            task_index=task_index,
            compress_fun=compress_fun
        )
        
        t1 = time.time()
        attributions = attributer.compute(
            x=x_mut,
            x_ref=x_ref,
            save_window=None,
            batch_size=16,  # ignored for DeepSHAP
            gpu=gpu
        )
        t2 = time.time() - t1
        print('Attribution time:', t2)
    else:
        # Use unified Attributer for other methods
        # Determine compression function based on task type
        if task_type == 'profile':
            compress_fun = Attributer.bpnet_profile
        elif task_type == 'counts':
            compress_fun = Attributer.bpnet_counts

        # =============================================================================
        # Compute attribution maps
        # =============================================================================
        attributer = Attributer(
            model,
            method=attribution_method,
            task_index=task_index,
            compress_fun=compress_fun
        )

        # Note: DeepSHAP processes sequences one at a time (no batch mode)
        # Other methods (saliency, smoothgrad, intgrad, ism) can use batch processing
        t1 = time.time()
        attributions = attributer.compute(
            x=x_mut,
            x_ref=x_ref,
            save_window=None,
            batch_size=16,  # ignored for DeepSHAP
            gpu=gpu,
            snv_window=[map_start, map_end] if attribution_method == 'ism' else None
        )
        t2 = time.time() - t1
        print('Attribution time:', t2)

    if save_data:
        np.save(os.path.join(save_path, f'attributions_{attribution_method}.npy'), attributions)

# Render logo of attribution map for reference sequence
if render_logos is True:
    reference_logo = BatchLogo(attributions[ref_index:ref_index+1, map_start:map_end],
    #reference_logo = BatchLogo(attributions[ref_index:ref_index+1, map_start+map_crop[0]:map_start+map_crop[1]],
        alphabet=alphabet,
        font_name='Arial Rounded MT Bold',
        fade_below=0.5,
        shade_below=0.5,
        width=0.9,
        figsize=[20,1.5],
        center_values=True,
        batch_size=1
    )

    reference_logo.process_all()

    fig, ax = reference_logo.draw_single(
        ref_index,
        border=False
    )
    if save_logos:
        fig.savefig(os.path.join(save_path, 'reference_logo.png'), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if 1: # crop attribution maps to focus on specific region of PPIF locus, enabling higher resolution analysis
    attributions = attributions[:,map_start+map_crop[0]:map_start+map_crop[1],:]
    mave_df['Sequence'] = mave_df['Sequence'].str[map_start+map_crop[0]:map_start+map_crop[1]]

# =============================================================================
# SEAM API
# Cluster attribution maps directly using Hierarchical Clustering
# =============================================================================
if load_previous_linkage is False:
    clusterer = Clusterer(
        attributions,
        gpu=gpu
    )

    # Perform hierarchical clustering directly on attribution maps
    link_method = 'ward'
    linkage = clusterer.cluster(
        method='hierarchical',
        link_method=link_method
    )

    if save_data:
        np.save(os.path.join(save_path, f'hierarchical_linkage_{link_method}_{attribution_method}.npy'), linkage)

labels_n, cut_level = clusterer.get_cluster_labels(
    linkage,
    criterion='maxclust',
    n_clusters=n_clusters
)

# Plot dendrogram to visualize hierarchy
if 0:
    clusterer.plot_dendrogram(
        linkage,
        cut_level=cut_level,
        figsize=(15, 10),
        leaf_rotation=90,
        leaf_font_size=8,
        save_path=save_path_figs,
        dpi=dpi
    )

# =============================================================================
# SEAM API
# Generate meta-explanations and related statistics
# =============================================================================
sort_method = 'median' # sort clusters by median DNN prediction (default)

# Initialize MetaExplainer
meta = MetaExplainer(
    clusterer=clusterer,
    mave_df=mave_df,
    attributions=attributions,
    sort_method=sort_method,
    ref_idx=0,
    mut_rate=mut_rate
)

# Plot boxplot distribution for DNN predictions associated with each cluster
pred_boxplots = meta.plot_cluster_stats(
    plot_type='box',
    metric='prediction',
    show_ref=True,
    show_fliers=False,
    compact=True,
    save_path=save_path_figs,
    dpi=dpi
)

# Plot occupancy distribution for attribution maps associated with each cluster
pred_boxplots = meta.plot_cluster_stats(
    plot_type='bar',
    metric='occupancy',
    show_ref=True,
    show_fliers=False,
    save_path=save_path_figs,
    dpi=dpi
)

# Generate Mechanism Summary Matrix (MSM)
msm = meta.generate_msm(
    gpu=gpu
)

# Plot MSM with different options
view_window = None

meta.plot_msm(column='Entropy',
    square_cells=True,
    view_window=view_window,
    save_path=save_path_figs,
    dpi=dpi
)

meta.plot_msm(column='Reference',
    square_cells=True,
    view_window=view_window,
    save_path=save_path_figs,
    dpi=dpi
)

# =============================================================================
# SEAM API
# Plot meta-attribution maps for each cluster
# =============================================================================
# Generate attribution logos
if render_logos is True:
    logo_type = 'average' # {average, pwm, enrichment}

    meta_logos = meta.generate_logos(logo_type=logo_type,
        background_separation=False,
        font_name='Arial Rounded MT Bold',
        fade_below=0.5,
        shade_below=0.5,
        width=0.9,
        center_values=True
    )

    if save_logos is True:
        save_path_logos = os.path.join(save_path, 'cluster_logos')
        if not os.path.exists(save_path_logos):
            os.makedirs(save_path_logos)
        save_path_logos_average = os.path.join(save_path_logos, logo_type)
        if not os.path.exists(save_path_logos_average):
            os.makedirs(save_path_logos_average)

    for cluster_index in tqdm(range(n_clusters), desc='Generating logos'):
        fig, ax = meta_logos.draw_single(
            cluster_index,
            fixed_ylim=True, # fixed y-axis limits as defined over all cluster-averaged logos
            figsize=(20, 1.5),
            border=False,
            view_window=view_window
        )
        if save_logos:
            fig.savefig(os.path.join(save_path_logos_average, 'cluster_%s.png' % cluster_index), facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    # Generate variability logo, representing the overlap of all averaged cluster logos
    fig, ax = meta_logos.draw_variability_logo(
        figsize=(20,1.5),
        view_window=view_window,
    )
    if save_logos:
        fig.savefig(os.path.join(save_path_logos_average, '_variability_logo.png'), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# =============================================================================
# SEAM API
# Background separation
# =============================================================================
# Plot variation in attribution values across attribution maps for each nucleotide position
meta.plot_attribution_variation(
    scope='clusters',  # {'all', 'clusters'}
    metric='std',  # {'std', 'var'}
    save_path=save_path_figs,
    dpi=dpi
)

# Separate individual clusters from average background over clusters
background_multiplier = 0.5  # default threshold factor for background separation

# View clusters without background
meta_logos_no_bg = meta.generate_logos(
    logo_type='average',
    background_separation=True,
    mut_rate=mut_rate,
    entropy_multiplier=background_multiplier,
    adaptive_background_scaling=True,
    figsize=(20, 1.5),
    font_name='Arial Rounded MT Bold',
    fade_below=0.5,
    shade_below=0.5,
    width=0.9,
    center_values=True
)

if save_logos is True:
    save_path_logos_no_bg = os.path.join(save_path_logos, '%s_no_bg' % logo_type)
    if not os.path.exists(save_path_logos_no_bg):
        os.makedirs(save_path_logos_no_bg)

for cluster_index in tqdm(range(n_clusters), desc='Generating logos'):
    fig, ax = meta_logos_no_bg.draw_single(
        cluster_index,
        fixed_ylim=True, # fixed y-axis limits as defined over all cluster-averaged logos
        figsize=(20, 1.5),
        border=False,
        view_window=view_window
    )
    if save_logos:
        fig.savefig(os.path.join(save_path_logos_no_bg, 'cluster_%s.png' % cluster_index), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# View average background over all clusters
average_background_logo = BatchLogo(
    meta.background[np.newaxis, :, :],
    alphabet=meta.alphabet,
    figsize=[20, 1.5],
    batch_size=1,
    font_name='Arial Rounded MT Bold',
    fade_below=0.5,
    shade_below=0.5,
    width=0.9,
    center_values=True
)
average_background_logo.process_all()

if render_logos is True:
    if sort_method is not None:
        ref_cluster = meta.membership_df.loc[ref_index, 'Cluster_Sorted']
    else:
        ref_cluster = meta.membership_df.loc[ref_index, 'Cluster']

    # Compare reference map with and without noise reduction and background separation
    # Reference logo
    reference_logo = BatchLogo(attributions[ref_index:ref_index+1],  # already-cropped attributions
        alphabet=alphabet,
        font_name='Arial Rounded MT Bold',
        fade_below=0.5,
        shade_below=0.5,
        width=0.9,
        figsize=[20,1.5],
        center_values=True,
        batch_size=1
    )

    reference_logo.process_all()

    fig, ax = reference_logo.draw_single(
        0,
        fixed_ylim=False,
        figsize=(20,1.5),
        border=False,
        view_window=view_window
        )
    if save_logos:
        fig.savefig(os.path.join(save_path_logos, '1_reference_logo.png'), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Reference cluster: noise reduction via averaging
    # Note: membership of the reference sequence to a cluster can be less reliable for more-variable Saliency maps and/or when using a small num_clusters
    fig, ax = meta_logos.draw_single(
        ref_cluster,
        fixed_ylim=False,
        figsize=(20,1.5),
        border=False,
        view_window=view_window
        )
    if save_logos:
        fig.savefig(os.path.join(save_path_logos, '2_reference_cluster.png'), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Background for WT cluster
    background_logo = BatchLogo(
        meta.cluster_backgrounds[ref_cluster:ref_cluster+1],
        alphabet=meta.alphabet,
        figsize=(20,1.5),
        batch_size=1,
        font_name='Arial Rounded MT Bold',
        fade_below=0.5,
        shade_below=0.5,
        width=0.9,
        center_values=True
    )
    background_logo.process_all()
    
    fig, ax = background_logo.draw_single(
        0,
        fixed_ylim=False,
        border=False,
        view_window=view_window
    )
    if save_logos:
        fig.savefig(os.path.join(save_path_logos, '3_bg_for_wt_cluster.png'), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Reference cluster: noise reduction and background separation
    fig, ax = meta_logos_no_bg.draw_single(
        ref_cluster,
        fixed_ylim=False,
        figsize=(20,1.5),
        border=False,
        view_window=view_window
        )
    if save_logos:
        fig.savefig(os.path.join(save_path_logos, '4_reference_cluster_no_bg.png'), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Background averaged over all clusters
    fig, ax = average_background_logo.draw_single(
        0,
        fixed_ylim=False,
        figsize=(20,1.5),
        border=False,
        view_window=view_window
        )
    if save_logos:
        fig.savefig(os.path.join(save_path_logos, '5_average_background.png'), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Save average background data to numpy array
if save_data:
    np.save(os.path.join(save_path, 'average_background.npy'), meta.background)
    
    if sort_method is not None:
        ref_cluster = meta.membership_df.loc[ref_index, 'Cluster_Sorted']
    else:
        ref_cluster = meta.membership_df.loc[ref_index, 'Cluster']
    
    # Calculate reference cluster-averaged attribution matrix
    ref_cluster_avg = np.mean(meta.get_cluster_maps(ref_cluster), axis=0)
    np.save(os.path.join(save_path, 'average_reference_cluster.npy'), ref_cluster_avg)
    
    # Average FG (foreground = average reference cluster - average background)
    foreground = ref_cluster_avg - meta.background
    np.save(os.path.join(save_path, 'average_foreground.npy'), foreground)
    
    # MSM data as CSV
    msm.to_csv(os.path.join(save_path, 'msm.csv'), index=False)