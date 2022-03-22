import tensorflow as tf

############################################################################
# Settings
FLAGS = tf.compat.v1.flags.FLAGS
############################################################################
tf.compat.v1.flags.DEFINE_string('run_name', 'weight_equal', 
	"""experiment name""")

tf.compat.v1.flags.DEFINE_float('learning_rate', 0.01,
	"""Learning rate""")

tf.compat.v1.flags.DEFINE_list('layer_size',[16],
	"""Layer dimension for graphs""")


tf.compat.v1.flags.DEFINE_float('decay_rate', 0.9, 
	"""Decay rate of the learning rate""")

tf.compat.v1.flags.DEFINE_integer('decay_steps', 10000, 
	"""Decay steps of the learning rate""")

tf.compat.v1.flags.DEFINE_integer('batch_size', 2,  #2
	"""Batch size""")

tf.compat.v1.flags.DEFINE_float('regu_scale', 0.001, #0.0001
	"""Scale for the regulizer""")

tf.compat.v1.flags.DEFINE_float('regulizer_scale', 1e-05, # -4 -6 -8
	"""Scale for the regulizer""")

tf.compat.v1.flags.DEFINE_integer('patience',30000,  # relax to
	"""Batch size""")

tf.compat.v1.flags.DEFINE_integer('max_train_steps',5, #100000
	"""The number of trained epoches""")

tf.compat.v1.flags.DEFINE_integer('train_steps_for_report', 1000, #141
	"""The number of trained epoches""")

tf.compat.v1.flags.DEFINE_integer('report_freq', 1000,
	"""Save the model after a fixed number of epoches""")

############################################################################
# Data
tf.compat.v1.flags.DEFINE_integer('num_classes', 2, 
	"""The number of labels """)

tf.compat.v1.flags.DEFINE_string('log_directory', '../../../../Models/Yufeng/EHR_evolve/',
	"""The output diretory for model saving""")


tf.compat.v1.flags.DEFINE_string('data_dir', '../../../../Data/Processed/Yufeng/Graph_tfrecords/',
	"""The graph data""")