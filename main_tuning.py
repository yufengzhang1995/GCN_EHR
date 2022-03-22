import os
import time
from flags import FLAGS
import main_train

tune = 'No_tune_cv'
layer_number = 1

if tune == 'No_tune_cv':
    checkpoint_root = os.path.join(FLAGS['log_directory'].value, 'CV_only')
    if not os.path.isdir(checkpoint_root):
      os.mkdir(checkpoint_root)
    data_root = FLAGS['data_dir'].value

    for fold in range(0,3):
        run_name = 'Learning_rate_{}_layer_size_{}_layer_num_{}'.format(FLAGS['learning_rate'].value, FLAGS['layer_size'].value[0],layer_number)
        run_id = 'c{}_{}_{}'.format(fold, run_name, time.strftime("%b_%d_%H_%M_%S", time.localtime()))
        print('Start to train {}'.format(run_id))
        checkpoint_path = os.path.join(checkpoint_root, run_id)
        train_dir = set([os.path.join(data_root, 'fold_{}'.format(x)) for x in range(3)])
        val_dir = os.path.join(data_root, 'fold_{}'.format(fold))
        train_dir.remove(val_dir)
        train = main_train.Train(n_classes = FLAGS['num_classes'].value, 
                  batch_size = FLAGS['batch_size'].value, 
                  layer_size = FLAGS['layer_size'].value,
                  learning_rate = FLAGS['learning_rate'].value,
                  decay_steps = FLAGS['decay_steps'].value,
                  decay_rate = FLAGS['decay_rate'].value,
                  regu_scale = FLAGS['regulizer_scale'].value) 
        train.train(train_dir, val_dir, checkpoint_path)