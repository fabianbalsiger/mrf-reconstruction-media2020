import argparse
import logging
import os

import pymia.deeplearning.tensorflow.logging as tflog
import tensorflow as tf

import mrf.configuration.config as cfg
import mrf.data.handler as hdlr
import mrf.data.split as split
import mrf.model.factory as mdl
import mrf.utilities.filesystem as fs
import mrf.utilities.logging as log
import mrf.utilities.seeding as seed
import mrf.utilities.training as train


def main(config_file: str):
    config = cfg.load(config_file, cfg.Configuration)

    # set up directories and logging
    model_dir, result_dir = fs.prepare_directories(config_file, cfg.Configuration, lambda: fs.get_directory_name(config))
    config.model_dir = model_dir
    config.result_dir = result_dir

    log.setup(os.path.join(config.model_dir, 'logging.log'))
    logging.info(config)

    # set seed before model instantiation
    logging.info(f'Set seed to {config.seed}')
    seed.set_seed(config.seed, config.cudnn_determinism)

    # load train and valid subjects from split file (also test but it is unused)
    subjects_train, subjects_valid, subjects_test = split.load_split(config.split_file)
    logging.info(f'Train subjects: {subjects_train}')
    logging.info(f'Valid subjects: {subjects_valid}')

    # set up data handling
    data_handler = hdlr.MRFDataHandler(config, subjects_train, subjects_valid, subjects_test, False,
                                       padding_size=mdl.get_padding_size(config))

    # extract a sample for model initialization
    data_handler.dataset.set_extractor(data_handler.extractor_train)
    sample = data_handler.dataset[0]

    with tf.Session() as sess:
        model = mdl.get_model(config)(sess, sample, config)
        logging.info(f'Number of parameters: {model.get_number_parameters()}')

        logger = tflog.TensorFlowLogger(config.model_dir, sess,
                                        model.epoch_summaries(),
                                        model.batch_summaries(),
                                        model.visualization_summaries())

        # trainer = train.AssemblingTesterTensorFlow(data_handler, logger, config, model, sess)  # use this class to test the pipeline
        trainer = train.MRFTrainer(data_handler, logger, config, model, sess)

        tf.get_default_graph().finalize()  # to ensure that no ops are added during training, which would lead to
        # a growing graph
        trainer.train()
        logger.close()


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Spatially Regularized Parametric Map Reconstruction for Fast Magnetic Resonance Fingerprinting')

    parser.add_argument(
        '--config_file',
        type=str,
        default='./config.json',
        help='Path to the configuration file.'
    )

    args = parser.parse_args()
    main(args.config_file)
