import numpy as np
import pandas as pd
import os
import tensorflow as tf

from flags import FLAGS
import network
import data_flow

class link_predict_tasker():
    def __init__(self,model_name,data_dir,subset):
        super(link_predict_tasker, self).__init__()
        self.model_name = model_name
        self.test_ds = data_flow.input_fn(data_dir, subset, batch_size = 1)
    
    
    def read_model_path(self):
        # ckpt_dir = os.path.join(FLAGS['log_directory'].value,self.model_name)
        ckpt_dir = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Models/Yufeng/EHR_evolve/CV_only/c1_Learning_rate_0.01_layer_size_16_layer_num_1_Mar_19_04_33_33/ckpt-8'
        print(ckpt_dir)


        model = network.GCN_Pool(sizes = [16], n_classes = 2) 
        ckpt = tf.train.Checkpoint(model = model)

        if 'ckpt' in ckpt_dir and os.path.exists(ckpt_dir + '.index'):
            ckpt.restore(ckpt_dir).expect_partial()
        elif os.path.isdir(ckpt_dir):
            ckpt.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()
        else:
            raise ValueError
        print(model.trainable_weights)

        for layer in model.layers:
            print("===== LAYER: ", layer.name, " =====")
            if layer.get_weights() != []:
                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]
                print("weights:")
                print(weights)
                print("biases:")
                print(biases)
            else:
                print("weights: ", [])

        # layer_name = []
        # for layer in model.layers:
        #     layer_name.append(layer.name)
        #     print(layer.name, layer)
        #     print(model.get_layer(layer.name).weights)
        # f_w = model.get_layer(layer_name[0]).weights
        # print(f_w)
        # if len(f_w)!=0:
        #     W = f_w[0].numpy()
        #     print(W.shape)

    def get_representation(self,model):
        # for data, labels in self.test_ds:
        pass
data_root = FLAGS['data_dir'].value
data_dir = os.path.join(data_root,'cv_test')
task = link_predict_tasker(model_name = 'CV_only/c2_Learning_rate_0.01_layer_size_16_layer_num_1_Mar_19_15_20_12/ckpt-4',
                           data_dir = data_dir,subset = 'cv_test' )

task.read_model_path()       