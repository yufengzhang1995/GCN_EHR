from numpy import genfromtxt
import numpy as np
from scipy import sparse
import os

root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/'
tfrecord_root = os.path.join(root,'Data/Processed/Yufeng/Graph_tfrecords')



embedding = genfromtxt('test_sgns_embedding.csv', delimiter=',')


X = sparse.csr_matrix(embedding)
sparse.save_npz(os.path.join(tfrecord_root,'X.npz'),X) 
print('Save X')

#'/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Processed/Yufeng/Graph_tfrecords/X.npz'