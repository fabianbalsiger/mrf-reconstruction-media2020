import logging

import tensorflow as tf

import mrf.configuration.config as cfg
import mrf.model.algorithm as algorithm
import mrf.model.base as mdl_base


MODEL_MEDIA = 'media'


def layer(x, filters, kernel_size, dilation_rate, is_training_placeholder,
          dropout_p: float = 0, norm: str = 'none', activation='relu',
          layer_no: int = 1):
    x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                         padding='valid', activation=activation, name='layer{}_conv'.format(layer_no))

    layer_no += 1

    if dropout_p > 0:
        x = tf.layers.dropout(x, dropout_p, training=is_training_placeholder)

    if norm == 'bn':
        x = tf.layers.batch_normalization(x, training=is_training_placeholder)

    return x, layer_no


def temporal_block(x, n_layers: int, growth_rate, kernel_size, dilation_rate, is_training_placeholder,
                   dropout_p: float = 0, norm: str = 'none', activation='relu', layer_no: int = 1):
    for i in range(n_layers):
        out, layer_no = layer(x, growth_rate, kernel_size, dilation_rate, is_training_placeholder,
                              dropout_p, norm, activation, layer_no)
        x = tf.concat([x, out], axis=-1)

    return x, layer_no


def spatial_block(x, filters, dilation_rate, is_training_placeholder,
                  dropout_p: float = 0, norm: str = 'none', activation='relu',
                  layer_no: int = 1):
    return layer(x, filters, 3, dilation_rate, is_training_placeholder, dropout_p, norm, activation, layer_no)


class MEDIAModel(mdl_base.MRFModel):

    def __init__(self, session, sample: dict, config: cfg.Configuration):
        self.dropout_p = 0.0
        self.norm = 'bn'
        self.activation_fn = 'relu'

        self.no_nonlinearities = config.no_nonlinearities
        self.no_channels_last_spatial_block = config.no_channels_last_spatial_block
        self.no_channels_minimum_temporal_block = config.no_channels_minimum_temporal_block
        self.no_channels_decrease_spatial = config.no_channels_decrease_spatial
        self.no_channels_decrease_temporal = config.no_channels_decrease_temporal

        no_channels_input = sample['images'].shape[-2] * sample['images'].shape[-1]

        parameter_calculator = algorithm.Algorithm(config.receptive_field, config.no_parameters,
                                                   len(config.maps), no_channels_input,
                                                   self.no_channels_last_spatial_block,
                                                   self.no_channels_minimum_temporal_block,
                                                   self.no_channels_decrease_temporal,
                                                   self.no_channels_decrease_spatial,
                                                   self.no_nonlinearities)

        self.no_channels_temporal_blocks = parameter_calculator.get_no_channels_temporal_blocks()
        self.no_channels_spatial_blocks = parameter_calculator.get_no_channels_spatial_blocks()
        self.no_layers_temporal_blocks = parameter_calculator.get_no_layers_temporal_blocks()

        super().__init__(session, sample, config)

        logging.info('Initialized model with receptive field of {:d} and {:d} parameters'.format(
            config.receptive_field, config.no_parameters))
        logging.info('Channels per temporal blocks: [' + ', '.join(map(str, self.no_channels_temporal_blocks)) + ']')
        logging.info('Channels per spatial blocks: [' + ', '.join(map(str, self.no_channels_spatial_blocks)) + ']')
        logging.info('Layers per temporal block: [' + ', '.join(map(str, self.no_layers_temporal_blocks)) + ']')
        logging.info('Number of non-linearities: {:d}'.format(self.no_nonlinearities))
        logging.info('Number of parameters: {:d}'.format(
            parameter_calculator.no_parameters_difference + config.no_parameters))

    def inference(self, x) -> object:
        x = tf.reshape(self.x_placeholder, (-1,
                                            self.x_placeholder.shape[1],
                                            self.x_placeholder.shape[2],
                                            self.x_placeholder.shape[3] * self.x_placeholder.shape[4]))

        layer_no = 1  # for graph naming

        if not self.no_channels_spatial_blocks:
            # receptive field of 1 x 1
            x, layer_no = temporal_block(x, self.no_layers_temporal_blocks[0], self.no_channels_temporal_blocks[0], 1, 1,
                                         self.is_training_placeholder, self.dropout_p, self.norm,
                                         self.activation_fn,
                                         layer_no)

            x, layer_no = layer(x, self.no_channels_last_spatial_block, 1, 1,
                                self.is_training_placeholder, self.dropout_p, self.norm, self.activation_fn,
                                layer_no)
        else:
            for temporal_block_ch, spatial_block_ch, n_layers in zip(self.no_channels_temporal_blocks,
                                                                     self.no_channels_spatial_blocks,
                                                                     self.no_layers_temporal_blocks):
                x, layer_no = temporal_block(x, n_layers, temporal_block_ch, 1, 1,
                                             self.is_training_placeholder, self.dropout_p, self.norm,
                                             self.activation_fn,
                                             layer_no)

                x, layer_no = spatial_block(x, spatial_block_ch, 1,
                                            self.is_training_placeholder, self.dropout_p, self.norm, self.activation_fn,
                                            layer_no)

        # last layer with number of maps as output
        x, layer_no = layer(x, self.no_maps, 1, 1,
                            self.is_training_placeholder, self.dropout_p, self.norm, 'linear', layer_no)

        return tf.identity(x, name='network')
