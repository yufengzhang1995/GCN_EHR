

from tensorflow.keras.layers import Dense,Conv1D
from tensorflow.keras import Model
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import activations, regularizers, constraints, initializers
spdot = tf.sparse.sparse_dense_matmul
dot = tf.matmul

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import sklearn


import numpy as np
import pandas as pd
import os
import time
import scipy.sparse

from flags import FLAGS

import network 
import data_flow




class Train():
    def __init__(self, 
                n_classes, 
                batch_size, 
                layer_size,
                learning_rate, 
                decay_steps, 
                decay_rate,
                regu_scale=0):
        """Initialization. """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.regu_scale = regu_scale
        self.n_classes = n_classes
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.model = network.GCN_Pool(sizes = layer_size, n_classes = n_classes)  
    
    def _dataset_loading(self,train_dir, val_dir):
        self.train_ds = data_flow.input_fn(train_dir, 'train', batch_size = self.batch_size)   
        self.val_ds = data_flow.input_fn(val_dir, 'val', batch_size = self.batch_size)                                  
    
    def _convert_prediction_to_one_hot(self, prediction):
        segmat = tf.stack([prediction]*self.n_classes, axis=-1)
        ones_map = tf.ones(prediction.shape, dtype=tf.int64)
        lis = []
        for i in range(self.n_classes):
            lis.append(ones_map*i)
        index_map = tf.stack(lis, axis=-1)
        return segmat == index_map
            
    def _define_loss_and_metrics(self):    
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                0.1,
                decay_steps=10000,
                decay_rate=0.96,
                staircase=True)
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True) 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        self.train_regu_loss = tf.keras.metrics.Mean(name = 'regulizer_loss')
        self.train_auc = tf.keras.metrics.AUC(name='train_auc')
        self.test_loss = tf.keras.metrics.Mean(name = 'test_loss')
        self.test_auc = tf.keras.metrics.AUC(name='test_auc')

    @tf.function
    def _train_step(self, data, labels):
        with tf.GradientTape() as tape:

            predictions = self.model(data, training=True)
            y_pred = tf.nn.softmax(predictions)
            
            labels_reshaped = tf.cast(tf.reshape(labels, [-1, FLAGS['num_classes'].value]),tf.float32)
            weight = tf.linalg.tensor_diag(tf.constant([1.0,1.0]), name=None)
            
            preds_reshaped = tf.cast(tf.reshape(y_pred, [-1, FLAGS['num_classes'].value]),tf.float32)
            weighted_label = tf.matmul(labels_reshaped, weight)
            
            
            loss = self.loss_object(weighted_label, preds_reshaped)   
            regu_loss = tf.reduce_sum(self.model.losses)
            total_loss = loss + regu_loss         
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        preds = tf.argmax(predictions, axis=-1)
        preds = self._convert_prediction_to_one_hot(preds)


        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_regu_loss(regu_loss)
        self.train_auc(labels, y_pred)
        return y_pred
    
            
    @tf.function
    def _test_step(self, data, labels):
        predictions = self.model(data,training = False)
        y_pred = tf.nn.softmax(predictions)
        preds = tf.argmax(predictions, axis=-1)
        preds = self._convert_prediction_to_one_hot(preds)
        loss = self.loss_object(labels, y_pred)
        self.test_loss(loss)
        self.test_auc(labels, y_pred)
        return y_pred, preds
    

    def _reset_states_for_train(self):
        self.train_loss.reset_states()
        self.train_auc.reset_states()
        self.train_regu_loss.reset_states()


    def _reset_states_for_val(self):
        self.test_loss.reset_states()
        self.test_auc.reset_states()
         

    def _write_tf_summary(self, subset, step, data = None):
        with self.summary_writer.as_default():
            if subset == 'train':
                tf.summary.scalar(f'{subset}/loss', self.train_loss.result(), step=step)
                tf.summary.scalar(f'{subset}/auc', self.train_auc.result(), step=step)
                tf.summary.scalar(f'{subset}/regu_loss', self.train_regu_loss.result(), step=step)

                    
            elif subset == "val" or subset == 'test' :
                tf.summary.scalar(f'{subset}/loss', self.test_loss.result(), step=step)
                tf.summary.scalar(f'{subset}/auc', self.test_auc.result(), step=step)

    def _validation(self,step):
        self._reset_states_for_val()
        
        y_pred_ls = []
        preds_ls = []
        labels_ls = []
        
        for data, labels in self.val_ds:
            y_pred,preds = self._test_step(data, labels) 
            y_pred_ls.append(y_pred)
            preds_ls.append(preds)
            labels_ls.append(labels)
        y_pred_ls = np.array(y_pred_ls).reshape((-1, FLAGS['num_classes'].value))
        preds_ls = np.array(preds_ls).reshape((-1, FLAGS['num_classes'].value))
        labels_ls = np.array(labels_ls).reshape((-1, FLAGS['num_classes'].value))
        
        
        prediction = tf.argmax(preds_ls,axis = -1)
        annotation = tf.argmax(labels_ls,axis = -1)
        prediction_prob = y_pred_ls[:,1]
        n_samples = prediction.shape[0]
        # print('n_samples:',n_samples)
        # print('annotation:',annotation)

        acc = np.sum(prediction == annotation) / n_samples
        tp = np.sum(np.all([prediction == 1, annotation == 1], axis = 0))
        tn = np.sum(np.all([prediction == 0, annotation == 0], axis = 0))
        sum_p = np.sum(annotation == 1) 
        sum_pred_p = np.sum(prediction == 1) 
        sum_n = np.sum(annotation == 0) 

        sen = tp / sum_p
        prec = tp / sum_pred_p
        spe = tn / sum_n
        f1 = 2 * prec * sen / (prec + sen)
        # auc = sklearn.metrics.roc_auc_score(annotation, prediction_prob)
        ap = sklearn.metrics.average_precision_score(annotation, prediction_prob)
        # print('True positives:',sum_p)
        # print("For {}  dataset: acc is {:.4}, auc is {:.4}, sen is {:.4},  prec is {:.4}, spe is {:.4}, f1 is {:.4}, ap is {:.4}".format(subset,acc,auc,sen, prec, spe,f1,ap))
        # precision = precision_score(labels_ls, preds_ls, average='macro')
        auc = network.multiclass_roc_auc_score(labels_ls,preds_ls)
        # recall = recall_score(labels_ls, preds_ls, average='macro')
        # f1score = 2 * precision * recall/(precision + recall)
        # acc = accuracy_score(labels_ls, preds_ls)
        print("The result for Val is : recall is {}, precision is {}, accuracy is {}, F1 is {},AUC is {}".format(sen,prec,acc,f1,auc))

        self.auc = auc
        self.f1score = f1
        self._write_tf_summary('val', step)
             
    def train(self, train_dir, val_dir, checkpoint_path):

        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        self._dataset_loading(train_dir, val_dir)    
        self._define_loss_and_metrics()
        self._reset_states_for_train()
        ckpt = tf.train.Checkpoint(model = self.model, optimizer = self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = 3)
        self.summary_writer = tf.summary.create_file_writer(checkpoint_path)

        best_val_value = float('inf')
        patience = 0
        global_step = 0
        
        
        for data, labels in self.train_ds:
            preds = self._train_step(data, labels)
            
            if global_step%FLAGS['report_freq'].value == 0:
                self._write_tf_summary('train', global_step, data)
                self._validation(global_step)
                
                # print training process
                template = 'Steps {}, Loss: {:.4}, Regulizer Loss: {:.4},  AUC:{:.4}%, Val Loss: {:.4}, Val AUC: {:.4}%'
                print(template.format(global_step,
                            self.train_loss.result(),
                            self.train_regu_loss.result(),
                            self.train_auc.result()*100,
                            self.test_loss.result(),
                            self.test_auc.result()*100))
                
                metrics_value = -1*self.auc
                if metrics_value < best_val_value:
                    patience = 0
                    ckpt_manager.save()
                    best_val_value = metrics_value
                else:
                    patience += 1

                if patience > (FLAGS['patience'].value+FLAGS['decay_steps'].value)//FLAGS['report_freq'].value:
                    print("The val acc doesn't improved for {} steps. Early stopped.".format(FLAGS['patience'].value))
                    break

            if global_step%FLAGS['train_steps_for_report'].value == 0:
                self._reset_states_for_train()

            # Stopping criterion
            if global_step > FLAGS['max_train_steps'].value:
                print('Achived the maximal training steps.')
                break

            global_step += 1

        
    def evaluate(self, data_dir, subset, checkpoint_dir,return_prob = False):
        self.test_ds = data_flow.input_fn(data_dir, subset, batch_size = 1) 
        ckpt = tf.train.Checkpoint(model = self.model)
        ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        print(self.model.trainable_weights)

        prob,preds,labels_ls = self._predict()

        prediction = tf.argmax(preds,axis = -1)
        annotation = tf.argmax(labels_ls,axis = -1)
        prediction_prob = prob[:,1]
        n_samples = prediction.shape[0]
        print('n_samples:',n_samples)
        # print('annotation:',annotation)

        acc = np.sum(prediction == annotation) / n_samples
        tp = np.sum(np.all([prediction == 1, annotation == 1], axis = 0))
        tn = np.sum(np.all([prediction == 0, annotation == 0], axis = 0))
        sum_p = np.sum(annotation == 1) 
        sum_pred_p = np.sum(prediction == 1) 
        sum_n = np.sum(annotation == 0) 

        sen = tp / sum_p
        prec = tp / sum_pred_p
        spe = tn / sum_n
        f1 = 2 * prec * sen / (prec + sen)
        auc = sklearn.metrics.roc_auc_score(annotation, prediction_prob)
        ap = sklearn.metrics.average_precision_score(annotation, prediction_prob)
        print('True positives:',sum_p)
        print("For {}  dataset: acc is {:.4}, auc is {:.4}, sen is {:.4},  prec is {:.4}, spe is {:.4}, f1 is {:.4}, ap is {:.4}".format(subset,acc,auc,sen, prec, spe,f1,ap))

        return [acc,auc,sen, prec, spe,f1,ap]
        
        
        
        
        # auc = network.multiclass_roc_auc_score(labels_ls,preds)
        
        
        
        # precision = precision_score(tf.argmax(labels_ls,axis = -1), tf.argmax(preds,axis = -1))
        # recall = recall_scoretf.argmax(labels_ls,axis = -1), tf.argmax(preds,axis = -1))
        # f1score = 2 * precision * recall/(precision + recall)
        # acc = accuracy_score(labels_ls, preds)
        # print("The result is : recall is {}, precision is {}, accuracy is {}, F1 is {},AUC is {}".format(recall,precision,acc,f1score,auc))
        # return np.array([precision,recall,acc,f1score,auc])
          
    def _predict(self):
        y_pred_ls = []
        preds_ls = []
        labels_ls = []
        for data, labels in self.test_ds:
            
            predictions = self.model(data,training = False)
            y_pred = tf.nn.softmax(predictions)
            preds = tf.argmax(predictions, axis=-1)
            preds = self._convert_prediction_to_one_hot(preds)
            
            y_pred_ls.append(y_pred)
            preds_ls.append(preds)
            labels_ls.append(labels)
        
        y_pred_ls = np.array(y_pred_ls).reshape((-1, FLAGS['num_classes'].value))
        preds_ls = np.array(preds_ls).reshape((-1, FLAGS['num_classes'].value))
        labels_ls = np.array(labels_ls).reshape((-1, FLAGS['num_classes'].value))
        return y_pred_ls,preds_ls,labels_ls
    

if __name__ == '__main__':
    # Settings
    import tensorflow as tf
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    data_root = FLAGS['data_dir'].value
    train = Train(n_classes = FLAGS['num_classes'].value, 
                  batch_size = FLAGS['batch_size'].value, 
                  layer_size = FLAGS['layer_size'].value,
                  learning_rate = FLAGS['learning_rate'].value,
                  decay_steps = FLAGS['decay_steps'].value,
                  decay_rate = FLAGS['decay_rate'].value,
                  regu_scale = FLAGS['regulizer_scale'].value) 

    option = 2
    if option == 1:
        print('Training ')
        checkpoint_root = FLAGS['log_directory'].value
        if not os.path.isdir(checkpoint_root):
            os.mkdir(checkpoint_root)

        train_dir = os.path.join(data_root, 'train')
        val_dir = os.path.join(data_root, 'val')
        
        run_id = '{}_{}'.format(FLAGS.run_name, time.strftime("%b_%d_%H_%M_%S", time.localtime()))
        checkpoint_path = os.path.join(checkpoint_root, run_id)
        train.train(train_dir, val_dir, checkpoint_path)

    elif option == 2:
        # folder_list = ['regular_Mar_06_13_21_57']
        # folder_list = ['small_lr_Mar_06_22_00_45'] # in fact not small lr
        # folder_list = ['small_lr_Mar_06_16_52_26']
        # folder_list = ['weight_equal_Mar_15_11_10_41']
        folder_list = ['CV_only/c0_Learning_rate_0.01_layer_size_16_layer_num_1_Mar_18_12_31_23',]
                        # 'CV_only/c1_Learning_rate_0.01_layer_size_16_layer_num_1_Mar_19_04_33_33',
                        # 'CV_only/c2_Learning_rate_0.01_layer_size_16_layer_num_1_Mar_19_15_20_12'    ] 

        eval_ls = []
        print(folder_list)
        for i in range(3):
            checkpoint_path = os.path.join(FLAGS['log_directory'].value, folder_list[i])
        
        # data_set = ['test']
        # for i in data_set:
        
        #     data_dir = os.path.join(data_root,i)
        #     train.evaluate(data_dir, i, checkpoint_path)
        #     print(i)
        #     print('************')
            print(checkpoint_path)
            data_dir = os.path.join(data_root,'cv_test')
            test_metrics = train.evaluate(data_dir, 'test', checkpoint_path)
            eval_ls.append(test_metrics)
            print('test')
            print('************')
        eval_table = np.concatenate(eval_ls, axis=1)
        print(np.mean(eval_table,axis = 1))
        print(np.std(eval_table,axis = 1))

        

                
        
    