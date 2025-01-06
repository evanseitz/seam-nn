import sys, os
sys.dont_write_bytecode = True
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance


def arr2pd(x, alphabet=['A','C','G','T']):
    """Function to convert a Numpy array to Pandas dataframe with proper column headings.

    Parameters
    ----------
    x : numpy.ndarray
        One-hot encoding or attribution map (shape : (L,C)).
    alphabet : list
        The alphabet used to determine the C characters in the logo such that
        each entry is a string; e.g., ['A','C','G','T'] for DNA.

    Returns
    -------
    x : pandas.dataframe
        Dataframe corresponding to the input array.
    """
    labels = {}
    idx = 0
    for i in alphabet:
        labels[i] = x[:,idx]
        idx += 1
    x = pd.DataFrame.from_dict(labels, orient='index').T
    
    return x


def oh2seq(one_hot, alphabet=['A','C','G','T']):
    """Function to convert one-hot encoding to a sequence.

    Parameters
    ----------
    one_hot : numpy.ndarray
        Input one-hot encoding of sequence (shape : (L,C))
    alphabet : list
        The alphabet used to determine the C characters in the logo such that
        each entry is a string; e.g., ['A','C','G','T'] for DNA.

    Returns
    -------
    seq : string
        Input sequence with length L.
    """
    seq = []
    for i in range(np.shape(one_hot)[0]):
        for j in range(len(alphabet)):
            if one_hot[i][j] == 1:
                seq.append(alphabet[j])
    seq = ''.join(seq)
    return seq


def seq2oh(seq, alphabet=['A','C','G','T']):
    """Function to convert a sequence to one-hot encoding.

    Parameters
    ----------
    seq : string
        Input sequence with length L
    alphabet : list
        The alphabet used to determine the C characters in the logo such that
        each entry is a string; e.g., ['A','C','G','T'] for DNA.

    Returns
    -------
    one_hot : numpy.ndarray
        One-hot encoding corresponding to input sequence (shape : (L,C)).
    """
    L = len(seq)
    one_hot = np.zeros(shape=(L,len(alphabet)), dtype=np.float32)
    for idx, i in enumerate(seq):
        for jdx, j in enumerate(alphabet):
            if i == j:
                one_hot[idx,jdx] = 1
    return one_hot


def twohot2seq(two_hot):
    """Function to convert two-hot encoding to a DNA sequence.

    Parameters
    ----------
    two_hot : numpy.ndarray
        Input one-hot encoding of sequence (shape : (L,C))

    Returns
    -------
    seq : string
        Input sequence with length L.
    """
    seq = []
    for i in range(two_hot.shape[0]):

        if np.array_equal(two_hot[i,:], np.array([2, 0, 0, 0])):
            seq.append('A')
        elif np.array_equal(two_hot[i,:], np.array([0, 2, 0, 0])):
            seq.append('C')
        elif np.array_equal(two_hot[i,:], np.array([0, 0, 2, 0])):
            seq.append('G')
        elif np.array_equal(two_hot[i,:], np.array([0, 0, 0, 2])):
            seq.append('T')
        elif np.array_equal(two_hot[i,:], np.array([0, 0, 0, 0])):
            seq.append('N')
        elif np.array_equal(two_hot[i,:], np.array([1, 1, 0, 0])):
            seq.append('M')
        elif np.array_equal(two_hot[i,:], np.array([1, 0, 1, 0])):
            seq.append('R')
        elif np.array_equal(two_hot[i,:],np.array([1, 0, 0, 1])):
            seq.append('W')
        elif np.array_equal(two_hot[i,:], np.array([0, 1, 1, 0])):
            seq.append('S')
        elif np.array_equal(two_hot[i,:], np.array([0, 1, 0, 1])):
            seq.append('Y')
        elif np.array_equal(two_hot[i,:], np.array([0, 0, 1, 1])):
            seq.append('K')
    seq = ''.join(seq)
    return seq


def seq2twohot(seq):
    """Function to convert heterozygous DNA sequence to two-hot encoding.

    Parameters
    ----------
    seq : string
        Input sequence with length L.

    Returns
    -------
    one_hot : numpy.ndarray
        Input one-hot encoding of sequence (shape : (L,C))
    """
    seq_list = list(seq.upper()) # get sequence into an array
    # one hot the sequence
    encoding = {
        "A": np.array([2, 0, 0, 0]),
        "C": np.array([0, 2, 0, 0]),
        "G": np.array([0, 0, 2, 0]),
        "T": np.array([0, 0, 0, 2]),
        "N": np.array([0, 0, 0, 0]),
        "M": np.array([1, 1, 0, 0]),
        "R": np.array([1, 0, 1, 0]),
        "W": np.array([1, 0, 0, 1]),
        "S": np.array([0, 1, 1, 0]),
        "Y": np.array([0, 1, 0, 1]),
        "K": np.array([0, 0, 1, 1]),
    }
    one_hot = [encoding.get(seq, seq) for seq in seq_list]
    one_hot = np.array(one_hot)
    return one_hot


def hamming_distance(seq1, seq2):
    """Calculate the Hamming distance between two sequences."""
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


def xmut2pd(x_mut, y_mut, alphabet=['A','C','G','T'], reference=0, hamming=True, encoding=1, gpu=False, save_dir=None):
    """Function to convert a Numpy array of one-hot sequences (x_mut) and corresponding
    DNN scores (y_mut) into a Pandas dataframe with proper column headings.

    Parameters
    ----------
    x_mut : numpy.ndarray
        One-hot encoding of ensemble of mutagenized sequences  (shape : (N,L,C)).
    y_mut : numpy.ndarray
        Ensemble of scalar-valued scores predicted for each of the elements in x_mut (shape : N)
    alphabet : list
        The alphabet used to determine the C characters in the logo such that
        each entry is a string; e.g., ['A','C','G','T'] for DNA.
    reference : {int>0, str, False}
        If not False, create an additional column for the Hamming distance between each sequence and the reference sequence
        An integer value specifies the index of the reference sequence in x_mut.
        A string of length L can alternatively be used to specify a new reference sequence; e.g., 'AATACGC...'
    encoding : int {1,2}
        Choice of either one-hot or two-hot encoding, with the latter used for encoding heterozygosity.
    gpu : boole
        If true, convert entire matrix at once (~twice as fast as standard approach if running on GPUs)
    save_dir : str
        Directory for saving figures to file.

    Returns
    -------
    mave_df : pandas.dataframe
        Dataframe corresponding to the input arrays.
    """

    ### ZULU TBD: 
        # columns:
            # GIA scores (needs background)
            # task/class
        # header:
            # num_classes
            # motif name
            # motif length
            # mut type: {local, r=0.15}, etc.
            # crop (Boole)

    N = x_mut.shape[0]

    if type(reference) == int:
        if encoding == 1:
            wt_full = oh2seq(x_mut[reference], alphabet=['A','C','G','T'])
        elif encoding == 2:
            wt_full = twohot2seq(x_mut[reference])
        mave_df = pd.DataFrame(columns = ['DNN', 'Hamming', 'Sequence',], index=range(N))
    elif type(reference) == str:
        wt_full = reference
        mave_df = pd.DataFrame(columns = ['DNN', 'Hamming', 'Sequence'], index=range(N))
    else:
        wt_full = None
        mave_df = pd.DataFrame(columns = ['DNN', 'Sequence'], index=range(N))

    mave_df['DNN'] = y_mut

    if encoding == 1:
        if gpu is False:
            alphabet_dict = {}
            idx = 0
            for i in range(len(alphabet)):
                alphabet_dict[i] = alphabet[i]
                idx += 1
            for i in range(N): #standard approach
                seq_index = np.argmax(x_mut[i,:,:], axis=1)
                seq = []
                for s in seq_index:
                    seq.append(alphabet_dict[s])
                seq = ''.join(seq)
                mave_df.at[i, 'Sequence'] = seq
        elif gpu is True:
            seq_index_all = np.argmax(x_mut, axis=-1)
            num2alpha = dict(zip(range(0, len(alphabet)), alphabet))
            seq_vector_all = np.vectorize(num2alpha.get)(seq_index_all)
            seq_list_all = seq_vector_all.tolist()
            dictionary_list = []
            for i in range(0, N, 1):
                dictionary_data = {0: ''.join(seq_list_all[i])}
                dictionary_list.append(dictionary_data)
            mave_df['Sequence'] = pd.DataFrame.from_dict(dictionary_list)
    elif encoding == 2: # ZULU optimize this
        for i in tqdm(range(N), desc='Encoding'):
            mave_df.at[i, 'Sequence'] = twohot2seq(x_mut[i])

    if wt_full is not None and hamming is True:
        for i, x in tqdm(enumerate(mave_df['Sequence']), desc='Hamming'):
            d = round(distance.hamming(list(wt_full), list(x)) * len(wt_full))
            mave_df.at[i, 'Hamming'] = d

    if save_dir is not None:
        mave_df.to_csv(os.path.join(save_dir, 'mave.csv'), index=False)

    return mave_df



if __name__ == "__main__":
    py_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(py_dir)

    mode = 'quantity'
    #mode = 'profile'

    dir_name = 'examples/examples_clipnet/outputs_local_NOTCH1/heterozygous_pt01/%s_nfolds9' % mode
    save_dir = os.path.join(parent_dir, dir_name)
 
    x_mut = np.load(os.path.join(parent_dir, dir_name + '/archives/x_mut.npy'))
    y_mut = np.load(os.path.join(parent_dir, dir_name + '/archives/y_mut.npy'))

    mave_df = xmut2pd(x_mut, y_mut, alphabet=['A','C','G','T','N','M','R','W','S','Y','K'], hamming=True, encoding=2, save_dir=save_dir)