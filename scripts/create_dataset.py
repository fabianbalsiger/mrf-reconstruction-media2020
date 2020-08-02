"""Creates a dataset, which is used to train the convolutional neural network.

We can only provide one image slice of one subject as data, since the data is private and subject to patient confidentiality.
To mimic the dataset creation as close as possible, from this image slice, we create eight volumetric subjects. These subjects
will then be loaded from the filesystem, such as you would load your own data.
"""
import argparse
import glob
import os
import pickle
import typing

import numpy as np
import pymia.data as pymia_data
import pymia.data.conversion as pymia_conv
import pymia.data.creation as pymia_crt
import pymia.data.definition as pymia_def
import pymia.data.transformation as pymia_tfm

import mrf.data.definition as defs
import mrf.data.transform as tfm


def create_sample_data(dir: str, no_subjects: int = 8, no_image_slices: int = 5):

    # we can only provide one image slice, to mimic the use data, we tile the image slice to obtain an image volume
    def tile(data: np.ndarray):
        return np.tile(data, (no_image_slices, ) + (1, ) * data.ndim)

    # load the measured fingerprint data of one slice
    # the shape is X x Y x T x 2 with spatial size X = Y = 350, T = 175 temporal frames, and real and imaginary parts
    fingerprints = tile(np.load(os.path.join(dir, 'fingerprints.npy')))

    # load the parametric maps reconstructed by dictionary matching
    map_ff = tile(np.load(os.path.join(dir, 'map_ff.npy')))  # the fat fraction (FF) map
    map_t1h2o = tile(np.load(os.path.join(dir, 'map_t1h2o.npy')))  # the T1 water (T1H2O) map
    map_t1fat = tile(np.load(os.path.join(dir, 'map_t1fat.npy')))  # the T1 fat (T1fat) map
    map_deltaf = tile(np.load(os.path.join(dir, 'map_deltaf.npy')))  # the off-resonance (deltaf) map
    map_b1 = tile(np.load(os.path.join(dir, 'map_b1.npy')))  # the flip angle efficacy (B1) map

    # load the mask for the parametric maps
    mask_fg = tile(np.load(os.path.join(dir, 'mask_fg.npy')))  # foreground mask
    mask_roi = tile(np.load(os.path.join(dir, 'mask_roi.npy')))  # manually segmented regions of interest (ROIs) of the major muscles for evaluation
    mask_t1h2o = tile(np.load(os.path.join(dir, 'mask_t1h2o.npy')))  # thresholded foreground mask for the T1H2O map
    mask_roi_t1h2o = tile(np.load(os.path.join(dir, 'mask_roi_t1h2o.npy')))  # thresholded ROI mask for the T1H2O map

    for i in range(no_subjects):
        subject = 'Subject_{}'.format(i)
        subject_path = os.path.join(dir, subject)
        os.makedirs(subject_path, exist_ok=True)

        np.save(os.path.join(subject_path, 'fingerprints.npy'), fingerprints)
        np.save(os.path.join(subject_path, 'map_ff.npy'), map_ff)
        np.save(os.path.join(subject_path, 'map_t1h2o.npy'), map_t1h2o)
        np.save(os.path.join(subject_path, 'map_t1fat.npy'), map_t1fat)
        np.save(os.path.join(subject_path, 'map_deltaf.npy'), map_deltaf)
        np.save(os.path.join(subject_path, 'map_b1.npy'), map_b1)
        np.save(os.path.join(subject_path, 'mask_fg.npy'), mask_fg)
        np.save(os.path.join(subject_path, 'mask_roi.npy'), mask_roi)
        np.save(os.path.join(subject_path, 'mask_t1h2o.npy'), mask_t1h2o)
        np.save(os.path.join(subject_path, 'mask_roi_t1h2o.npy'), mask_roi_t1h2o)


class Collector:
    """Collects the subjects and the data of each subject."""

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.subject_files = []

        self._collect()

    def get_subject_files(self) -> typing.List[pymia_data.SubjectFile]:
        return self.subject_files

    def _collect(self):
        self.subject_files.clear()

        subject_dirs = glob.glob(os.path.join(self.root_dir, '*'))
        subject_dirs = list(filter(lambda path: os.path.basename(path).lower().startswith('subject') and os.path.isdir(path),
                                   subject_dirs))
        subject_dirs.sort(key=lambda path: os.path.basename(path))

        # for each subject
        for subject_dir in subject_dirs:
            subject = os.path.basename(subject_dir)

            images = {defs.ID_DATA: os.path.join(subject_dir, 'fingerprints.npy')}
            labels = {defs.ID_MAP_FF: os.path.join(subject_dir, 'map_ff.npy'),
                      defs.ID_MAP_T1H2O: os.path.join(subject_dir, 'map_t1h2o.npy'),
                      defs.ID_MAP_T1FAT: os.path.join(subject_dir, 'map_t1fat.npy'),
                      defs.ID_MAP_DF: os.path.join(subject_dir, 'map_deltaf.npy'),
                      defs.ID_MAP_B1: os.path.join(subject_dir, 'map_b1.npy')}
            mask_fg = {defs.ID_MASK_FG: os.path.join(subject_dir, 'mask_fg.npy')}
            mask_t1h2o = {defs.ID_MASK_T1H2O: os.path.join(subject_dir, 'mask_t1h2o.npy')}
            mask_roi = {defs.ID_MASK_ROI: os.path.join(subject_dir, 'mask_roi.npy')}
            mask_roi_t1h2o = {defs.ID_MASK_ROI_T1H2O: os.path.join(subject_dir, 'mask_roi_t1h2o.npy')}

            sf = pymia_data.SubjectFile(subject, images=images, labels=labels,
                                        mask_fg=mask_fg, mask_t1h2o=mask_t1h2o,
                                        mask_roi=mask_roi, mask_roi_t1h2o=mask_roi_t1h2o)

            self.subject_files.append(sf)


class WriteNormalizationCallback(pymia_crt.Callback):
    """Writes the normalization values to the dataset."""

    def __init__(self, writer: pymia_crt.Writer, min_: dict, max_: dict) -> None:
        self.writer = writer
        self.min_ = min_
        self.max_ = max_

    def on_end(self, params: dict):
        # write min and max
        for key, value in self.min_.items():
            self.writer.write('norm/min/{}'.format(key), value, np.float32)

        for key, value in self.max_.items():
            self.writer.write('norm/max/{}'.format(key), value, np.float32)


class LoadData(pymia_crt.Load):
    """Loads the data from the filesystem."""

    def __init__(self, properties):
        self.properties = properties

    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) -> \
            typing.Tuple[np.ndarray, typing.Union[pymia_conv.ImageProperties, None]]:
        return np.load(file_name), self.properties


def concat(data: typing.List[np.ndarray]) -> np.ndarray:
    # in the dataset, the last dimension are the channels. For the fingerprints, this is 2 (real and imaginary) and
    # e.g. for the parametric maps 5. As the fingerprints are already in the correct format, we do not stack them!
    if data[0].ndim == 5:
        # no need to stack mrf data
        return data[0]
    return np.stack(data, axis=-1)


def main(hdf_file: str, data_dir: str):
    if os.path.exists(hdf_file):
        raise RuntimeError('Dataset file "{}" does already exist'.format(hdf_file))

    # we provide data from one image slice of one subject
    # from this data, we create "artificial" subjects to mimic the dataset creation process as close as possible
    # note that all subjects will be exactly of the same data
    create_sample_data(data_dir, no_subjects=8)

    # collect the data
    collector = Collector(data_dir)
    subjects = collector.get_subject_files()
    for subject in subjects:
        print(subject.subject)

    # get image properties
    with open(os.path.join(data_dir, 'properties.pickle'), 'rb') as handle:
        properties = pickle.load(handle)

    # get min and max of parametric maps for normalization
    # usually, the min and max are determined by the dictionary entries
    with open(os.path.join(data_dir, 'norm.pickle'), 'rb') as handle:
        norm_data = pickle.load(handle)
    min_ = norm_data['min']
    max_ = norm_data['max']

    # we create a HDF5 dataset from the subjects, for easier data access using the pymia package
    with pymia_crt.Hdf5Writer(hdf_file) as writer:
        callbacks = pymia_crt.get_default_callbacks(writer)
        callbacks.callbacks.append(WriteNormalizationCallback(writer, min_, max_))

        transform = pymia_tfm.ComposeTransform([
            tfm.MaskedParametricMapNormalization(min_, max_, defs.ID_MASK_FG),  # normalize parametric maps
            pymia_tfm.IntensityNormalization(loop_axis=4, entries=(pymia_def.KEY_IMAGES, )),  # normalize fingerprints
        ])

        traverser = pymia_crt.SubjectFileTraverser()
        traverser.traverse(subjects, callback=callbacks, load=LoadData(properties), transform=transform, concat_fn=concat)


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Dataset creation')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../data/data.h5',
        help='Path to the dataset.'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data',
        help='Path to the data directory.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.data_dir)
