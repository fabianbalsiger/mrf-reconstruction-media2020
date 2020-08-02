import abc

import pymia.data.definition as pymia_def
import pymia.deeplearning.tensorflow.model as mdl
import tensorflow as tf

import mrf.configuration.config as cfg
import mrf.data.definition as defs


class MRFModel(mdl.TensorFlowModel, abc.ABC):

    def placeholders(self, x_shape: tuple, y_shape: tuple):
        self.x_placeholder = tf.placeholder(tf.float32, x_shape, pymia_def.KEY_IMAGES)
        self.y_placeholder = tf.placeholder(tf.float32, y_shape, pymia_def.KEY_LABELS)
        self.is_training_placeholder = tf.placeholder(tf.bool,
                                                      name='is_training')  # True if training phase, otherwise False
        # define mask placeholders
        mask_shape = y_shape[:-1] + (1, )
        self.mask_fg_placeholder = tf.placeholder(tf.float32, mask_shape, defs.ID_MASK_FG)
        self.mask_t1h2o_placeholder = tf.placeholder(tf.float32, mask_shape, defs.ID_MASK_T1H2O)

    def __init__(self, session, sample: dict, config: cfg.Configuration):
        self.maps = config.maps
        self.no_maps = len(self.maps)
        self.learning_rate = config.learning_rate

        self.mask_fg_placeholder = None
        self.mask_t1h2o_placeholder = None

        # call base class constructor after initializing variables used by the implemented abstract functions
        super().__init__(session, config.model_dir,
                         x_shape=(None,) + sample[pymia_def.KEY_IMAGES].shape[1:],
                         y_shape=(None,) + sample[pymia_def.KEY_LABELS].shape[1:])

        self.add_summaries()

    def epoch_summaries(self) -> list:
        # not used
        return []

    def batch_summaries(self):
        # not used
        return []

    def visualization_summaries(self):
        # not used
        return [], [], [], []

    def add_summaries(self):
        tf.summary.scalar('train/loss', self.loss)

    def loss_function(self, prediction, label=None, **kwargs):
        # exclude background voxels
        mask = tf.concat([self.mask_fg_placeholder] * self.no_maps, -1)
        y = self.y_placeholder * mask
        y_predicted = self.network * mask

        loss = tf.losses.mean_squared_error(y, y_predicted)
        loss = tf.identity(loss, name='loss')
        return loss

    def optimize(self, **kwargs):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # add extra operations of the graph to the optimizer
        # e.g. this is used for batch normalization
        # see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        return train_op
