import argparse
import distutils
import glob
import os

import pymia.data.definition as pymia_def
import pymia.data.transformation as pymia_tfm
import tensorflow as tf

import mrf.configuration.config as cfg
import mrf.data.definition as defs
import mrf.data.handler as hdlr
import mrf.data.split as split
import mrf.data.transform as tfm
import mrf.model.factory as mdl
import mrf.utilities.testing as test


def permute(tester, result_dir: str, temporal_dim: int):
    for temporal_idx in range(temporal_dim):
        print('Permute temporal dimension {}...'.format(temporal_idx))
        # modify extraction transform for permutation experiment
        tester.data_handler.extraction_transform_valid = pymia_tfm.ComposeTransform([
            pymia_tfm.Squeeze(entries=(pymia_def.KEY_IMAGES, pymia_def.KEY_LABELS,
                                       defs.ID_MASK_FG, defs.ID_MASK_T1H2O), squeeze_axis=0),
            tfm.TemporalPermutation(temporal_idx=temporal_idx, entries=(pymia_def.KEY_IMAGES,))
        ])
        tester.result_dir = os.path.join(result_dir, 'permutation{}'.format(temporal_idx))
        tester.predict()


def main(model_dir: str, result_dir: str, do_permutation: bool):
    if not os.path.isdir(model_dir):
        raise RuntimeError('Model dir "{}" does not exist'.format(model_dir))

    config = cfg.load(glob.glob(os.path.join(model_dir, '*config*.json'))[0], cfg.Configuration)
    split_file = glob.glob(os.path.join(model_dir, '*split*.json'))[0]

    os.makedirs(result_dir, exist_ok=True)

    # load train, valid, and test subjects from split file
    subjects_train, subjects_valid, subjects_test = split.load_split(split_file)
    print('Test subjects:', subjects_test)

    # set up data handling
    data_handler = hdlr.MRFDataHandler(config, subjects_train, subjects_valid, subjects_test, False,
                                       padding_size=mdl.get_padding_size(config))

    # extract a sample for model initialization
    data_handler.dataset.set_extractor(data_handler.extractor_train)
    sample = data_handler.dataset[0]

    with tf.Session() as sess:
        model = mdl.get_model(config)(sess, sample, config)
        tester = test.MRFTensorFlowTester(data_handler, model, model_dir, result_dir, config, sess)
        tester.load(os.path.join(model_dir, config.best_model_file_name))

        print('Predict...')
        tester.predict()
        if do_permutation:
            permute(tester, result_dir, sample[pymia_def.KEY_IMAGES].shape[-2])


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Spatially Regularized Parametric Map Reconstruction for Fast Magnetic Resonance Fingerprinting')

    parser.add_argument(
        '--model_dir',
        type=str,
        default='../model',
        help='Path to the model directory.'
    )

    parser.add_argument(
        '--result_dir',
        type=str,
        default=None,
        help='Path to the results directory.'
    )

    parser.add_argument(
        '--do_permutation',
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help='True runs a permutation experiment.'
    )

    args = parser.parse_args()
    main(args.model_dir, args.result_dir, args.do_permutation)
