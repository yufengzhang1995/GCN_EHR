from scipy.sparse.csgraph import connected_components
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import networkx as nx
import numpy as np
import os
import pickle
import scipy.sparse as sp
import tensorflow as tf
from collections import defaultdict

def sp_matrix_to_sp_tensor(M):
    """ Convert a sparse matrix to a SparseTensor
    Parameters
    ----------
    M: a scipy.sparse matrix
    Returns
    -------
    X: a tf.SparseTensor
    Notes
    -----
    Also see tf.SparseTensor, scipy.sparse.csr_matrix
    H. J. @ 2019-02-12
    """
    if not isinstance(M, sp.csr.csr_matrix):
        M = M.tocsr()
    row, col = M.nonzero()
    X = tf.SparseTensor(np.mat([row, col]).T, M.data, M.shape)
    X = tf.cast(X, tf.float32)
    return X

def pad_with_last_val(vect,k):

    pad = tf.ones(k - vect.shape[0],
                         dtype=tf.int64) * vect[-1]                   
    vect = tf.concat([vect,pad],0)
    return vect



def data_split():
    pass