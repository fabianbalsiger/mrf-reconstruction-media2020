import logging
import math
import os

import numpy as np
import pymia.data.assembler as pymia_asmbl
import pymia.data.conversion as pymia_conv
import pymia.data.definition as pymia_def
import pymia.deeplearning.data_handler as hdlr
import pymia.deeplearning.training as train
import pymia.deeplearning.tensorflow.training as tftrain
import pymia.deeplearning.tensorflow.logging as tflog
import pymia.deeplearning.tensorflow.model as mdl
import pymia.evaluation.evaluator as pymia_eval
import SimpleITK as sitk
import tensorflow as tf

import mrf.configuration.config as cfg
import mrf.data.definition as defs
import mrf.utilities.assembler as asmbl
import mrf.utilities.metric as metric
import mrf.utilities.normalization as norm
import mrf.utilities.evaluation as eval
import mrf.utilities.filesystem as fs
import mrf.utilities.plt_qualitative as plt
import mrf.utilities.plt_quantitative as stat


def validate_on_subject(self: train.Trainer, subject_assembler: pymia_asmbl.SubjectAssembler,
                        config: cfg.Configuration, is_training: bool) -> float:

    # prepare filesystem and evaluator
    if self.current_epoch % self.save_validation_nth_epoch == 0:
        epoch_result_dir = fs.prepare_epoch_result_directory(config.result_dir, self.current_epoch)
        epoch_csv_file = os.path.join(
            epoch_result_dir,
            '{}_{}{}.csv'.format(os.path.basename(config.result_dir), self.current_epoch,
                                 '_train' if is_training else ''))
        epoch_csv_roi_file = os.path.join(
            epoch_result_dir,
            '{}_ROI_{}{}.csv'.format(os.path.basename(config.result_dir), self.current_epoch,
                                     '_train' if is_training else ''))
        epoch_csv_roi_summary_file = os.path.join(
            epoch_result_dir,
            '{}_ROI_SUMMARY_{}{}.csv'.format(os.path.basename(config.result_dir), self.current_epoch,
                                             '_train' if is_training else ''))
        epoch_txt_file = os.path.join(
            epoch_result_dir,
            '{}_{}{}.txt'.format(os.path.basename(config.result_dir), self.current_epoch,
                                 '_train' if is_training else ''))
        if is_training:
            writers = [pymia_eval.CSVEvaluatorWriter(epoch_csv_file)]
        else:
            writers = [pymia_eval.ConsoleEvaluatorWriter(5), pymia_eval.CSVEvaluatorWriter(epoch_csv_file)]
        evaluator = eval.Evaluator(writers, metric.get_metrics(), config.maps)
        evaluator_roi = eval.ROIEvaluator([pymia_eval.CSVEvaluatorWriter(epoch_csv_roi_file)], config.maps,
                                          config.label_file_dir, config.label_files)
    elif is_training:
        return float(-np.inf)
    else:
        epoch_result_dir = None
        epoch_csv_file = None
        epoch_csv_roi_summary_file = None
        epoch_txt_file = None
        evaluator = eval.Evaluator([pymia_eval.ConsoleEvaluatorWriter(5)], metric.get_metrics(), config.maps)
        evaluator_roi = eval.ROIEvaluator([], config.maps, config.label_file_dir, config.label_files)

    if not is_training:
        print('Epoch {}, {} s:'.format(self._get_current_epoch_formatted(), self.epoch_duration))

    # loop over all subjects
    for subject_idx in list(subject_assembler.predictions.keys()):
        subject_data = self.data_handler.dataset.direct_extract(self.data_handler.extractor_test, subject_idx)
        subject_name = subject_data['subject']

        # for voxel-wise dataset, we need to reshape the voxel-wise data to the original shape
        for k, v in subject_data.items():
            if isinstance(v, np.ndarray):
                subject_data[k] = np.reshape(v, subject_data[pymia_def.KEY_SHAPE] + (v.shape[-1],))

        # rescale and mask reference maps (clipping will have no influence)
        maps = norm.process(subject_data[pymia_def.KEY_LABELS], subject_data[defs.ID_MASK_FG],
                            subject_data[defs.KEY_NORM], config.maps)

        # rescale, clip, and mask prediction
        prediction = subject_assembler.get_assembled_subject(subject_idx)
        prediction = np.reshape(prediction, subject_data[pymia_def.KEY_SHAPE] + (prediction.shape[-1],))
        prediction = norm.process(prediction, subject_data[defs.ID_MASK_FG], subject_data[defs.KEY_NORM], config.maps)

        # evaluate
        evaluator.evaluate(prediction, maps, {'FG': subject_data[defs.ID_MASK_FG],
                                              'T1H2O': subject_data[defs.ID_MASK_T1H2O]},
                           subject_name)
        roi_masks = {'FG': subject_data[defs.ID_MASK_ROI], 'T1H2O': subject_data[defs.ID_MASK_ROI_T1H2O]}
        evaluator_roi.evaluate(prediction, roi_masks, {'FG': subject_data[defs.ID_MASK_FG],
                                                       'T1H2O': subject_data[defs.ID_MASK_FG]},
                               subject_name)

        # Save predictions as SimpleITK images and plot slice images
        if not is_training and (self.current_epoch % self.save_validation_nth_epoch == 0):
            subject_results = os.path.join(epoch_result_dir, subject_name)
            os.makedirs(subject_results, exist_ok=True)
            plotter = plt.QualitativePlotter(subject_results, 2, 'png')

            for map_idx, map_name in enumerate(config.maps):
                map_name_short = map_name.replace('map', '')
                # save predicted maps
                prediction_image = pymia_conv.NumpySimpleITKImageBridge.convert(prediction[..., map_idx],
                                                                                subject_data[pymia_def.KEY_PROPERTIES])
                sitk.WriteImage(prediction_image,
                                os.path.join(subject_results, '{}_{}.mha'.format(subject_name, map_name_short)),
                                True)

                plotter.plot(subject_name, map_name, prediction[..., map_idx], maps[..., map_idx],
                             subject_data[defs.ID_MASK_T1H2O] if map_name == defs.ID_MAP_T1H2O
                             else subject_data[defs.ID_MASK_FG])

    evaluator.write()
    evaluator_roi.write()

    # log to TensorBoard
    summaries = evaluator.get_summaries()
    for result in summaries:
        self.logger.log_scalar('{}/{}-MEAN'.format(result.map_, result.metric), result.mean, self.current_epoch,
                               is_training)
        self.logger.log_scalar('{}/{}-STD'.format(result.map_, result.metric), result.std, self.current_epoch,
                               is_training)

    roi_calculator = eval.ROICalculator(config.maps)
    roi_results = roi_calculator.calculate(evaluator_roi.results, config.roi_reference_file)
    scores = []
    for roi_result in roi_results:
        self.logger.log_scalar('{}/{}'.format(roi_result.map_, roi_result.metric), roi_result.mean, self.current_epoch,
                               is_training)
        scores.append(roi_result.mean)
        summaries.append(roi_result)

    print('Aggregated {} results (epoch {}):'.format('training' if is_training else 'validation',
                                                     self._get_current_epoch_formatted()))

    if self.current_epoch % self.save_validation_nth_epoch == 0:
        eval.SummaryResultWriter(epoch_txt_file).write(summaries)
        stat.QuantitativePlotter(epoch_result_dir).plot(epoch_csv_file,
                                                        'summary_train' if is_training else 'summary',
                                                        False if is_training else True)
        stat.QuantitativeROIPlotter(epoch_result_dir, config.maps).plot(epoch_csv_roi_file,
                                                                        config.roi_reference_file,
                                                                        'train' if is_training else '')
        roi_calculator.save_summary(evaluator_roi.results, config.roi_reference_file, epoch_csv_roi_summary_file)
    else:
        eval.SummaryResultWriter().write(summaries)

    return float(np.mean(scores)) if not is_training else -math.inf


class MRFTrainer(tftrain.TensorFlowTrainer):

    def __init__(self, data_handler: hdlr.DataHandler, logger: tflog.TensorFlowLogger,
                 config: cfg.Configuration, model: mdl.TensorFlowModel, session: tf.Session):
        super().__init__(data_handler, logger, config, model, session)
        self.config = config

    def init_subject_assembler(self) -> pymia_asmbl.Assembler:
        return asmbl.init_subject_assembler()

    def validate_on_subject(self, subject_assembler: pymia_asmbl.SubjectAssembler, is_training: bool) -> float:
        return validate_on_subject(self, subject_assembler, self.config, is_training)

    def batch_to_feed_dict(self, batch: dict, is_training: bool):
        feed_dict = {self.model.x_placeholder: np.stack(batch[pymia_def.KEY_IMAGES], axis=0),
                     self.model.y_placeholder: np.stack(batch[pymia_def.KEY_LABELS], axis=0),
                     self.model.mask_fg_placeholder: np.stack(batch[defs.ID_MASK_FG], axis=0),
                     self.model.mask_t1h2o_placeholder: np.stack(batch[defs.ID_MASK_T1H2O], axis=0),
                     self.model.is_training_placeholder: is_training}

        return feed_dict


class AssemblingTesterTensorFlow(MRFTrainer):
    """Use this class to test the training/validation without a network. The metrics should have the maximum values.

    Beware of possible data augmentation during testing!
    """

    def _get_labels_in_tensorflow_format(self, batch):
        feed_dict = self.batch_to_feed_dict(batch, True)
        return feed_dict[self.model.y_placeholder]

    def train_batch(self, idx, batch: dict):
        if idx % self.log_nth_batch == 0:
            logging.info('Epoch {}, batch {}/{:d}: loss={:5f}'
                         .format(self._get_current_epoch_formatted(),
                                 self._get_batch_index_formatted(idx),
                                 len(self.data_handler.loader_train),
                                 0.0))
        return self._get_labels_in_tensorflow_format(batch), 0.0

    def validate_batch(self, idx: int, batch: dict) -> (np.ndarray, float):
        return self._get_labels_in_tensorflow_format(batch), 0.0
