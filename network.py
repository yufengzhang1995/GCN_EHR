from tensorflow.keras.layers import Dense,Conv1D
from tensorflow.keras import Model
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import activations, regularizers, constraints, initializers
spdot = tf.sparse.sparse_dense_matmul
dot = tf.matmul

import scipy.sparse
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import sklearn

def sp_matrix_to_sp_tensor(M):
    if not isinstance(M, sp.csr.csr_matrix):
        M = M.tocsr()
    M.sum_duplicates()
    M.eliminate_zeros()
    row, col = M.nonzero()
    X = tf.SparseTensor(np.mat([row, col]).T, M.data, M.shape)
    X = tf.cast(X, tf.float32)
    return X


class GCNConv(layers.Layer):
    def __init__(self,
                 units,
                 use_bias = True,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer = None,
                 kernel_constraint = None,
                 bias_initializer = 'zeros',
                 bias_regularizer = None,
                 bias_constraint = None,
                 activity_regularizer = None,
                 **kwargs):

        self.units = units
        self.activation = tf.keras.layers.Activation('sigmoid')
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(GCNConv, self).__init__()
    def build(self, input_shape):
        """ GCN has two inputs : [shape(An), shape(X)]
        """
        fdim = input_shape[1][-1]  # feature dim 200

        if not hasattr(self, 'weight'):
            self.weight = self.add_weight(name="weight",
                                        shape=(fdim, self.units),
                                        initializer=self.kernel_initializer,
                                        constraint=self.kernel_constraint,
                                        trainable=True)
        if self.use_bias:
            if not hasattr(self, 'bias'):
                self.bias = self.add_weight(name="bias",
                                            shape=(self.units,),
                                            initializer=self.bias_initializer,
                                            constraint=self.bias_constraint,
                                            trainable=True)
        super(GCNConv, self).build(input_shape)

    def call(self, inputs):
        """ GCN has two inputs : [An, X]
        """
        self.An = inputs[0]
        self.X = inputs[1]

        if isinstance(self.X, tf.SparseTensor):
            h = spdot(self.X, self.weight)
        else:
            h = dot(self.X, self.weight)
        output = dot(self.An, h)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation:
            output = self.activation(output)
        return output


class GCN_Pool(Model):
    def __init__(self, sizes, n_classes = 2):
        super(GCN_Pool,self).__init__()
        self.layer_sizes = sizes
        self.layer1 = GCNConv(self.layer_sizes[0], activation='relu')
        # self.layer2 = GCNConv(self.layer_sizes[1])

        self.global_average_layer = layers.GlobalAveragePooling1D()
        
        self.dropout = layers.Dropout(0.5)
        self.prediction_layer = layers.Dense(n_classes)
        self.X = sp_matrix_to_sp_tensor(scipy.sparse.load_npz('/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Processed/Yufeng/Graph_tfrecords/X.npz'))

    def call(self, An):

        h1 = self.layer1([An,self.X])
        # h2 = self.layer2([An, h1])
        # if len(h1.shape) != 3:
        #     h2 = h2[tf.newaxis,:,:]  
        # h3 = self.global_average_layer(h2)
        h3 = self.global_average_layer(h1)
        x = self.dropout(h3)
        x = self.prediction_layer(x)
        return x


class MLP(Model):
    def __init__(self, sizes, n_classes = 2):
        super(MLP,self).__init__()












def multiclass_roc_auc_score(truth, pred):
    # lb = sklearn.preprocessing.LabelBinarizer()
    # lb.fit(truth)
    # truth = lb.transform(truth)
    # pred = lb.transform(pred)  
    # fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, scores)          
    return sklearn.metrics.roc_auc_score(truth, pred)





