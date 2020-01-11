import argparse
import glob
import os
import typing

import numpy as np
import pymia.data as pymia_data
import pymia.data.conversion as pymia_conv
import pymia.data.creation as pymia_crt
import pymia.data.definition as pymia_def
import pymia.data.transformation as pymia_tfm
import SimpleITK as sitk

import mrf.data.data as data
import mrf.data.transform as tfm


class Collector:

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.subject_files = []

        self._collect()

    def get_subject_files(self) -> typing.List[pymia_data.SubjectFile]:
        return self.subject_files

    def _collect(self):
        self.subject_files.clear()

        subject_dirs = glob.glob(os.path.join(self.root_dir, '*'))
        subject_dirs = list(filter(lambda path: os.path.basename(path).lower().startswith('subject')
                                                and os.path.isdir(path),
                                   subject_dirs))
        subject_dirs.sort(key=lambda path: os.path.basename(path))

        # for each subject
        for subject_dir in subject_dirs:
            subject = os.path.basename(subject_dir)

            images = {data.FileTypes.Data.name: os.path.join(subject_dir, 'MRFreal.mha')}
            labels = {data.FileTypes.T1H2Omap.name: os.path.join(subject_dir, 'T1H2O.mha'),
                      data.FileTypes.FFmap.name: os.path.join(subject_dir, 'FF.mha'),
                      data.FileTypes.B1map.name: os.path.join(subject_dir, 'B1.mha')}
            mask_fg = {data.FileTypes.ForegroundTissueMask.name: os.path.join(subject_dir, 'MASK_FG.mha')}
            mask_t1h2o = {data.FileTypes.T1H2OTissueMask.name: os.path.join(subject_dir, 'MASK_FG.mha')}
            mask_roi = {data.FileTypes.ROIMask.name: os.path.join(subject_dir, 'MASK_ROI.mha')}

            sf = pymia_data.SubjectFile(subject, images=images, labels=labels,
                                        mask_fg=mask_fg, mask_t1h2o=mask_t1h2o, mask_roi=mask_roi)

            self.subject_files.append(sf)


class WriteNormalizationCallback(pymia_crt.Callback):

    def __init__(self, writer: pymia_crt.Writer, min_: dict, max_: dict) -> None:
        self.writer = writer
        self.min_ = min_
        self.max_ = max_

    def on_start(self, params: dict):
        pass

    def on_end(self, params: dict):
        # write min and max
        for key, value in self.min_.items():
            self.writer.write('norm/min/{}'.format(key), value, np.float32)

        for key, value in self.max_.items():
            self.writer.write('norm/max/{}'.format(key), value, np.float32)

    def on_subject(self, params: dict):
        pass


def get_normalization_values(subjects: typing.List[pymia_data.SubjectFile], loader: pymia_crt.Load):
    """Calculates min and max of each map over all subjects. Excludes background values using the mask."""
    mins = {}
    maxs = {}

    for subject in subjects:

        maps = {img_key: loader(path, img_key, '', subject.subject)[0]
                for img_key, path in subject.categories[pymia_def.KEY_LABELS].entries.items()}
        mask_fg = loader(subject.categories[data.ID_MASK_FG].entries[data.FileTypes.ForegroundTissueMask.name],
                         data.FileTypes.ForegroundTissueMask.name,
                         '', subject.subject)[0]

        for map_name, map_ in maps.items():
            if map_name not in mins:
                mins[map_name] = []
                maxs[map_name] = []

            # mask needs to be binary. 1=foreground, 0=background.
            masked_map = np.ma.masked_array(map_, mask=mask_fg ^ 1)  # ensure min and max are not from background
            mins[map_name].append(masked_map.min())
            maxs[map_name].append(masked_map.max())

    min_ = {}
    max_ = {}
    for key in mins.keys():
        min_[key] = min(mins[key])
        max_[key] = max(maxs[key])

    return min_, max_


class LoadData(pymia_crt.Load):

    def __init__(self, ff_threshold: float = 0.7):
        # init some variables for faster loading
        self.ff_threshold = ff_threshold
        self.ff = None  # to create the mask_t1h2o

        self.mask_fg = None
        self.properties = None

    def _read_mask(self, file_name):
        img = sitk.ReadImage(file_name)
        np_data = sitk.GetArrayFromImage(img)
        # shape=(5, 350, 350) and dtype=uint8
        properties = pymia_conv.ImageProperties(img)
        return np_data, properties

    def _read_map(self, file_name):
        img = sitk.ReadImage(file_name)
        np_data = sitk.GetArrayFromImage(img)
        # shape=(5, 350, 350) and dtype=float32
        properties = pymia_conv.ImageProperties(img)
        return np_data, properties

    def _create_mask_t1h2o(self):
        mask_ff = self.ff.copy()
        mask_ff[self.ff < self.ff_threshold] = 1
        mask_ff[self.ff >= self.ff_threshold] = 0

        mask_t1h2o = mask_ff * self.mask_fg  # assumes that foreground mask is loaded
        return mask_t1h2o.astype(np.uint8)

    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) -> \
            typing.Tuple[np.ndarray, typing.Union[pymia_conv.ImageProperties, None]]:

        if id_ == data.FileTypes.Data.name:
            real = sitk.ReadImage(file_name)
            np_real = sitk.GetArrayFromImage(real)
            imag = sitk.ReadImage(file_name.replace('MRFreal', 'MRFimag'))
            np_imag = sitk.GetArrayFromImage(imag)

            # read image properties from a map
            _, self.properties = self._read_map(os.path.join(os.path.dirname(file_name), 'T1H2O.mha'))

            # shape=(5, 350, 350, 175)
            np_data = np.concatenate([np.expand_dims(np_real, -1),
                                      np.expand_dims(np_imag, -1)], axis=-1)
            return np_data, self.properties
        elif id_ == data.FileTypes.T1H2Omap.name:
            return self._read_map(file_name)
        elif id_ == data.FileTypes.FFmap.name:
            map_, properties = self._read_map(file_name)
            self.ff = map_
            return self.ff, properties
        elif id_ == data.FileTypes.B1map.name:
            return self._read_map(file_name)
        elif id_ == data.FileTypes.ForegroundTissueMask.name:
            mask, properties = self._read_mask(file_name)
            self.mask_fg = mask
            return self.mask_fg, properties
        elif id_ == data.FileTypes.T1H2OTissueMask.name:
            # this mask is not loaded but computed during data set creation
            # assumes that mask_fg is set!
            mask = self._create_mask_t1h2o()
            return mask, self.properties
        elif id_ == data.FileTypes.ROIMask.name:
            mask, _ = self._read_mask(file_name)
            return mask, self.properties
        else:
            raise ValueError('Unknown key')


def concat(data: typing.List[np.ndarray]) -> np.ndarray:
    if data[0].ndim == 5:
        # no need to stack mrf data
        return data[0]
    return np.stack(data, axis=-1)


def create_sample_data(dir: str, no_subjects: int = 8):
    def create_and_save_mrf(path, size, T):
        arr = np.random.rand(*size[::-1] + (T, ))
        img = sitk.GetImageFromArray(arr.astype(np.float32))
        sitk.WriteImage(img, path, True)

    def create_and_save_map(path, size, min_, max_):
        arr = np.random.rand(*size[::-1])
        arr = arr * (max_ - min_) + min_
        img = sitk.GetImageFromArray(arr.astype(np.float32))
        sitk.WriteImage(img, path, True)

    def create_and_save_mask(path, size, no_labels: int = 2):
        arr = np.random.randint(0, no_labels, size[::-1])
        img = sitk.GetImageFromArray(arr.astype(np.uint8))
        sitk.WriteImage(img, path, True)

    # MRF image series of size 350 x 350 x 5 x 175 = X x Y x Z x T
    image_shape = (350, 350, 5)
    no_temporal_frames = 175

    for i in range(no_subjects):
        subject = 'Subject_{}'.format(i)
        subject_path = os.path.join(dir, subject)
        os.makedirs(subject_path, exist_ok=True)

        # save MRF images real and imaginary separately (for simplicity)
        create_and_save_mrf(os.path.join(subject_path, 'MRFreal.mha'), image_shape, no_temporal_frames)
        create_and_save_mrf(os.path.join(subject_path, 'MRFimag.mha'), image_shape, no_temporal_frames)

        # save parametric maps
        create_and_save_map(os.path.join(subject_path, 'T1H2O.mha'), image_shape, 500, 2000)
        create_and_save_map(os.path.join(subject_path, 'FF.mha'), image_shape, 0, 1)
        create_and_save_map(os.path.join(subject_path, 'B1.mha'), image_shape, 0.5, 1)

        # save masks
        create_and_save_mask(os.path.join(subject_path, 'MASK_FG.mha'), image_shape)
        create_and_save_mask(os.path.join(subject_path, 'MASK_ROI.mha'), image_shape, 10)  # 9 ROIs (+ background9


def main(hdf_file: str, data_dir: str):
    if os.path.exists(hdf_file):
        raise RuntimeError('Dataset file "{}" does already exist'.format(hdf_file))

    # let's create some sample data
    np.random.seed(42)  # to have same sample data
    create_sample_data(data_dir, no_subjects=8)

    # collect the data
    collector = Collector(data_dir)
    subjects = collector.get_subject_files()
    for subject in subjects:
        print(subject.subject)

    # get the values for parametric map normalization
    min_, max_ = get_normalization_values(subjects, LoadData())

    with pymia_crt.Hdf5Writer(hdf_file) as writer:
        callbacks = pymia_crt.get_default_callbacks(writer)
        callbacks.callbacks.append(WriteNormalizationCallback(writer, min_, max_))

        transform = pymia_tfm.ComposeTransform([
            tfm.MRFMaskedLabelNormalization(min_, max_, data.ID_MASK_FG),
            pymia_tfm.IntensityNormalization(loop_axis=4, entries=(pymia_def.KEY_IMAGES, )),
        ])

        traverser = pymia_crt.SubjectFileTraverser()
        traverser.traverse(subjects, callback=callbacks,
                           load=LoadData(), transform=transform, concat_fn=concat)


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Dataset creation')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../data/data.h5',
        help='Path to the data set file.'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data',
        help='Path to the data directory.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.data_dir)
