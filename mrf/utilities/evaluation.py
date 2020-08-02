import os
import typing

import numpy as np
import pandas as pd
import pymia.evaluation.evaluator as pymia_eval
import pymia.evaluation.metric as pymia_metric
import scipy.stats
import SimpleITK as sitk

import mrf.data.definition as defs
import mrf.data.label as lbl


class Result:

    def __init__(self, subject: str, map_: str, metric: str, mask: str, value: float):
        self.subject = subject
        self.map_ = map_
        self.metric = metric
        self.mask = mask
        self.value = value

    def __eq__(self, other):
        return self.subject == other.subject and self.map_ == other.label and \
               self.metric == other.metric and self.mask == other.mask


class ROIResult(Result):

    def __init__(self, subject: str, map_: str, roi: str, metric: str, mask: str, slice: int, value: float):
        super().__init__(subject, map_, metric, mask, value)
        self.roi = roi
        self.slice = slice

    def __eq__(self, other):
        return self.subject == other.subject and self.map_ == other.label and self.roi == other.roi and  \
               self.metric == other.metric and self.mask == other.mask


class SummaryResult:

    def __init__(self, map_: str, metric: str, mask: str, mean: float, std: float):
        self.map_ = map_
        self.metric = metric
        self.mask = mask
        self.mean = mean
        self.std = std

    def __eq__(self, other):
        return self.map_ == other.label and self.metric == other.metric and self.mask == other.mask

    def __lt__(self, other):
        return self.map_[0] > other.map_[0] and self.metric[0] > other.metric[0]


def get_default_map_mask_combinations(maps: list):
    out = {}
    for map_ in maps:
        if map_ == defs.ID_MAP_T1H2O:
            out[map_.replace('map', '')] = 'T1H2O'
        elif map_ == defs.ID_MAP_T1FAT:
            out[map_.replace('map', '')] = 'FG'
        elif map_ == defs.ID_MAP_FF:
            out[map_.replace('map', '')] = 'FG'
        elif map_ == defs.ID_MAP_DF:
            out[map_.replace('map', '')] = 'FG'
        elif map_ == defs.ID_MAP_B1:
            out[map_.replace('map', '')] = 'FG'
        else:
            raise ValueError('Map "{}" not supported'.format(map_.replace('map', '')))

    return out


class Evaluator:

    def __init__(self, writers: typing.List[pymia_eval.IEvaluatorWriter], metrics: typing.List[pymia_metric.IMetric],
                 maps: list, map_mask_combinations_fn=get_default_map_mask_combinations):
        self.writers = writers
        self.metrics = metrics
        self.maps = maps
        # order of maps in configuration will correspond to order of array extraction in data loading, and therefore to
        # order of prediction

        self.map_idx_combinations = {map_.replace('map', ''): idx for idx, map_ in enumerate(self.maps)}
        self.map_mask_combinations = map_mask_combinations_fn(self.maps)
        self.header = []
        self.results = []
        self.results_for_writers = []
        self._write_header()

    def evaluate(self, maps_prediction: np.ndarray, maps_reference: np.ndarray, masks: dict, subject_id: str):
        """Evaluates the desired map-mask combinations on a predicted subject.

        Args:
            maps_prediction: The predicted maps. Size is (Z, Y, X, N) where N is the number of maps.
            maps_reference: The reference maps. Size is (Z, Y, X, N) where N is the number of maps.
            masks: A dict with various masks of size (Z, Y, X). The dict keys identify the masks.
            subject_id: The subject identification.
        """

        # for masking we use np.extract, which works only on 1-D arrays --> reshape
        maps_prediction_flattened = np.reshape(maps_prediction, (-1, maps_prediction.shape[-1]))
        maps_reference_flattened = np.reshape(maps_reference, (-1, maps_reference.shape[-1]))
        masks_flattened = {mask_key: np.reshape(mask, (-1, mask.shape[-1])) for mask_key, mask in masks.items()}

        for map_id, mask_id in self.map_mask_combinations.items():
            map_results = [subject_id, mask_id, map_id]

            # mask
            map_idx = self.map_idx_combinations[map_id]
            map_prediction = np.extract(masks_flattened[mask_id] == 1, maps_prediction_flattened[..., map_idx])
            map_reference = np.extract(masks_flattened[mask_id] == 1, maps_reference_flattened[..., map_idx])
            map_prediction_img = sitk.GetImageFromArray(maps_prediction[..., map_idx])
            map_reference_img = sitk.GetImageFromArray(maps_reference[..., map_idx])

            for metric in self.metrics:
                if isinstance(metric, pymia_metric.INumpyArrayMetric):
                    metric.ground_truth = map_reference
                    metric.segmentation = map_prediction
                elif isinstance(metric, pymia_metric.ISimpleITKImageMetric):
                    metric.ground_truth = map_reference_img
                    metric.segmentation = map_prediction_img
                else:
                    raise NotImplementedError('Only INumpyArrayMetric and ISimpleITKImageMetric implemented')

                result = metric.calculate()
                map_results += [result, ]
                self.results.append(Result(subject_id, map_id, metric.metric, mask_id, result))

            self.results_for_writers.append(map_results)

    def write(self):
        for writer in self.writers:
            writer.write(self.results_for_writers)

    def _write_header(self):
        self.header = ['ID', 'MASK', 'MAP'] + [metric.metric for metric in self.metrics]
        for writer in self.writers:
            writer.write_header(self.header)

    def get_summaries(self) -> typing.List[SummaryResult]:
        summaries = []

        for map_id, mask_id in self.map_mask_combinations.items():
            for metric in self.metrics:
                results = [result.value for result in self.results if
                           result.map_ == map_id and result.mask == mask_id and result.metric == metric.metric]
                summaries.append(SummaryResult(map_id, metric.metric, mask_id,
                                               float(np.mean(results)), float(np.std(results))))
        return summaries


class ROIEvaluator:

    def __init__(self, writers: typing.List[pymia_eval.IEvaluatorWriter], maps: tuple,
                 label_files_path: str, label_files: typing.List[str],
                 metrics: dict = dict(MEAN=np.mean, STD=np.std, MEDIAN=np.median),
                 map_mask_combinations_fn=get_default_map_mask_combinations):
        self.writers = writers
        self.metrics = metrics
        self.maps = maps
        # order of maps in configuration will correspond to order of array extraction in data loading, and therefore to
        # order of prediction

        self.label_files_path = label_files_path
        self.label_files = label_files
        self.labels = self._load_labels()

        self.map_idx_combinations = {map_.replace('map', ''): idx for idx, map_ in enumerate(self.maps)}
        self.map_mask_combinations = map_mask_combinations_fn(self.maps)
        self.header = []
        self.results = []
        self.results_for_writers = []
        self._write_header()

    def evaluate(self, maps_prediction: np.ndarray, roi_masks: dict, masks: dict, subject_id: str):
        """Evaluates the desired map-mask combinations on a predicted subject.

        Args:
            maps_prediction: The predicted maps. Size is (Z, Y, X, N) where N is the number of maps.
            roi_masks: A dict with various region of interest (ROI) mask. The dict keys identify the masks.
                Size is (Z, Y, X). The mask contains labels according to a label file.
            masks: A dict with various masks of size (Z, Y, X). The dict keys identify the masks.
            subject_id: The subject identification.
        """

        # get possible labels of the current region
        if defs.REGION_THIGH in subject_id:
            labels = self.labels[defs.REGION_THIGH]
        elif defs.REGION_LEG in subject_id:
            labels = self.labels[defs.REGION_LEG]
        elif 'Subject_' in subject_id:
            # fallback for dummy data
            labels = self.labels[defs.REGION_THIGH]
        else:
            raise ValueError('Unknown region of subject {}'.format(subject_id))

        for map_id, mask_id in self.map_mask_combinations.items():
            for slice_no in range(maps_prediction.shape[0]):
                # for masking we use np.extract, which works only on 1-D arrays --> reshape
                maps_prediction_flattened = np.reshape(maps_prediction[slice_no],
                                                       (-1, maps_prediction[slice_no].shape[-1]))
                roi_mask_flattened = {mask_key: np.reshape(mask[slice_no], (-1, mask[slice_no].shape[-1]))
                                      for mask_key, mask in roi_masks.items()}
                masks_flattened = {mask_key: np.reshape(mask[slice_no], (-1, mask[slice_no].shape[-1]))
                                   for mask_key, mask in masks.items()}

                # mask
                map_idx = self.map_idx_combinations[map_id]
                map_prediction = np.extract(masks_flattened[mask_id] == 1, maps_prediction_flattened[..., map_idx])
                roi_mask = np.extract(masks_flattened[mask_id] == 1, roi_mask_flattened[mask_id])

                for label in labels:
                    if label.value in roi_mask:
                        map_results = [subject_id, mask_id, map_id, label.label, '{}'.format(slice_no)]
                        map_prediction_by_roi = np.extract(roi_mask == label.value, map_prediction)
                        for metric_name, metric in self.metrics.items():
                            result = float(metric(map_prediction_by_roi))
                            map_results += [result, ]

                            self.results.append(ROIResult(subject_id, map_id, label.label, metric_name, mask_id,
                                                          slice_no, result))

                        self.results_for_writers.append(map_results)

    def write(self):
        for writer in self.writers:
            writer.write(self.results_for_writers)

    def _write_header(self):
        self.header = ['ID', 'MASK', 'MAP', 'ROI', 'SLICE'] + [metric for metric in self.metrics.keys()]
        for writer in self.writers:
            writer.write_header(self.header)

    def _load_labels(self):
        labels_dict = {}

        for file in self.label_files:
            file_path = os.path.join(self.label_files_path, file)
            labels = lbl.ITKSNAPLabelFileParser.parse(file_path)

            region = file.split('_')[1].split('.')[0].upper()
            if defs.REGION_THIGH in region:
                region = region[:-1]  # since it is thighS we remove the s
                labels = lbl.filter_labels_by_values(labels, lbl.get_thigh_label_values())
            elif defs.REGION_LEG in region:
                region = region[:-1]  # since it is legS we remove the s
                labels = lbl.filter_labels_by_values(labels, lbl.get_leg_label_values())
            else:
                raise ValueError('Unknown region "{}" of label file {}'.format(region, file_path))
            labels_dict[region] = labels

        return labels_dict


class ROICalculator:

    def __init__(self, maps: tuple = (defs.ID_MAP_T1H2O, defs.ID_MAP_T1FAT, defs.ID_MAP_FF,
                                      defs.ID_MAP_DF, defs.ID_MAP_B1)):
        self.maps = maps
        self.metrics = ['MEAN']

    def calculate(self, results: typing.List[ROIResult], csv_file_reference: str) -> typing.List[SummaryResult]:
        # convert to pandas
        prediction = pd.DataFrame(
            [[result.subject, result.mask, result.map_, result.roi, result.slice, result.value] for result in results
             if result.metric == 'MEAN'],
            columns=['ID', 'MASK', 'MAP', 'ROI', 'SLICE', 'MEAN'])
        reference = pd.read_csv(csv_file_reference, sep=';')
        return self._calculate(prediction, reference)

    def calculate_from_csv(self, csv_file_prediction: str, csv_file_reference: str) -> typing.List[SummaryResult]:
        prediction = pd.read_csv(csv_file_prediction, sep=';')
        reference = pd.read_csv(csv_file_reference, sep=';')
        return self._calculate(prediction, reference)

    def save_summary(self, results: typing.List[ROIResult], csv_file_reference: str, summary_csv_file_path: str):
        # convert to pandas
        prediction = pd.DataFrame(
            [[result.subject, result.mask, result.map_, result.roi, result.slice, result.value] for result in results
             if result.metric == 'MEAN'],
            columns=['ID', 'MASK', 'MAP', 'ROI', 'SLICE', 'MEAN'])
        reference = pd.read_csv(csv_file_reference, sep=';')
        df = pd.merge(prediction, reference, how='left', on=['ID', 'MASK', 'MAP', 'ROI', 'SLICE'], suffixes=('', '_REF'))
        for metric in self.metrics:
            df[metric + '_DIFF'] = df[metric] - df[metric + '_REF']
            df[metric + '_ABS_DIFF'] = abs(df[metric] - df[metric + '_REF'])
        df.to_csv(summary_csv_file_path, sep=';', index=False)

    def _calculate(self, prediction, reference):
        # join reference and prediction data frames such that we only have the predicted subjects
        df = pd.merge(prediction, reference, how='left', on=['ID', 'MASK', 'MAP', 'ROI', 'SLICE'], suffixes=('', '_REF'))

        results = []
        for map_ in self.maps:
            map_short = map_.replace('map', '')
            values = df[df['MAP'] == map_short]

            # not all map mask combinations have been calculated on CSV file generation
            if values.count().any() == 0:
                continue

            for metric in self.metrics:
                prediction_data = values[metric].values
                reference_data = values[metric + '_REF'].values
                _, _, r, _, _ = scipy.stats.linregress(prediction_data, reference_data)
                r2 = r**2

                results.append(SummaryResult(map_short, 'R2', 'ROI', r2, 0.0))

        return results


class SummaryResultWriter:

    def __init__(self, file_path: str = None, to_console: bool = True, precision: int = 3):
        self.file_path = file_path
        self.to_console = to_console
        self.precision = precision

    def write(self, data: typing.List[SummaryResult]):
        header = ['MAP', 'MASK', 'METRIC', 'RESULT']
        data = sorted(data)

        # we store the output data as list of list to nicely format the intends
        out_as_string = [header]
        for result in data:
            out_as_string.append([result.map_,
                                  result.mask,
                                  result.metric,
                                  '{0:.{2}f} ± {1:.{2}f}'.format(result.mean, result.std, self.precision)])

        # determine length of each column for output alignment
        lengths = np.array([list(map(len, row)) for row in out_as_string])
        lengths = lengths.max(0)
        lengths += (len(lengths) - 1) * [2] + [0, ]  # append two spaces except for last column

        # format for output alignment
        out = [['{0:<{1}}'.format(val, lengths[idx]) for idx, val in enumerate(line)] for line in out_as_string]

        to_print = '\n'.join(''.join(line) for line in out)
        if self.to_console:
            print(to_print)

        if self.file_path is not None:
            with open(self.file_path, 'w+') as file:
                file.write(to_print)
