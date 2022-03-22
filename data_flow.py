import tensorflow as tf
import glob
import functools
from flags import FLAGS 
import os
# import scipy.sparse
# from scipy.sparse import csr_matrix
# import scipy.sparse as sp
# import numpy as np

# def sp_matrix_to_sp_tensor(M):
#     if not isinstance(M, sp.csr.csr_matrix):
#         M = M.tocsr()
#     M.sum_duplicates()
#     M.eliminate_zeros()
#     row, col = M.nonzero()
#     X = tf.SparseTensor(np.mat([row, col]).T, M.data, M.shape)
#     X = tf.cast(X, tf.float32)
#     return X
class DataSet(object):
    def __init__(self, data_dir, subset = 'train'):
        self.data_dir = data_dir
        self.subset = subset

    def get_filenames(self):
        filenames = []
        if isinstance(self.data_dir, str):
            self.data_dir = [self.data_dir]
        for folder in self.data_dir:
            filenames += glob.glob(os.path.join(folder, '*tfrecord'))
        return filenames

    def parser(self, serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'graph/adj': tf.io.FixedLenFeature([3], tf.string),
                'graph/subject_id': tf.io.FixedLenFeature([], tf.string),
                'graph/label': tf.io.FixedLenFeature([], tf.int64)
            })
        features['graph/adj'] = tf.expand_dims(features['graph/adj'], axis=0)
        # adj = tf.io.deserialize_many_sparse(features['graph/adj'], dtype=tf.float32)
        adj = tf.squeeze(tf.sparse.to_dense(tf.io.deserialize_many_sparse(features['graph/adj'], dtype=tf.float32)))
        label = int(features['graph/label']) - 1
        label = tf.one_hot(label,depth = FLAGS['num_classes'].value)

        # x = sp_matrix_to_sp_tensor(scipy.sparse.load_npz('/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Processed/Yufeng/Graph_tfrecords/X.npz'))

        return adj,label


# build dataset
def input_fn(data_dir,subset, batch_size):
    data = DataSet(data_dir = data_dir, subset = subset)
    filenames = data.get_filenames()
    if subset == 'train':
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(filenames).repeat()
        dataset = dataset.shuffle(200, reshuffle_each_iteration = True)
    else:
        dataset = tf.data.TFRecordDataset(filenames)
    # Parse records.
    dataset = dataset.map(data.parser, num_parallel_calls = 4)
    dataset = dataset.batch(batch_size, drop_remainder = True)
    return dataset 