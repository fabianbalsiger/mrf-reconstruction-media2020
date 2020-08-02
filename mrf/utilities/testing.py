import os

import numpy as np
import pymia.data.assembler as pymia_asmbl
import pymia.data.conversion as pymia_conv
import pymia.data.definition as pymia_def
import pymia.deeplearning.data_handler as hdlr
import pymia.deeplearning.tensorflow.model as tfmdl
import pymia.deeplearning.tensorflow.testing as tftest
import pymia.deeplearning.testing as test
import pymia.evaluation.evaluator as pymia_eval
import SimpleITK as sitk
import tensorflow as tf

import mrf.configuration.config as cfg
import mrf.data.definition as defs
import mrf.utilities.assembler as asmbl
import mrf.utilities.metric as metric
import mrf.utilities.normalization as norm
import mrf.utilities.evaluation as eval
import mrf.utilities.plt_qualitative as plt
import mrf.utilities.plt_quantitative as stat


def process_predictions(self: test.Tester, subject_assembler: pymia_asmbl.SubjectAssembler, result_dir,
                        config: cfg.Configuration):

    os.makedirs(result_dir, exist_ok=True)
    csv_file = os.path.join(result_dir, 'RESULTS.csv')
    csv_roi_file = os.path.join(result_dir, 'RESULTS_ROI.csv')
    summary_file = os.path.join(result_dir, 'SUMMARY.txt')
    csv_roi_summary_file = os.path.join(result_dir, 'SUMMARY_ROI.csv')

    evaluator = eval.Evaluator([pymia_eval.CSVEvaluatorWriter(csv_file)], metric.get_metrics(), config.maps)
    evaluator_roi = eval.ROIEvaluator([pymia_eval.CSVEvaluatorWriter(csv_roi_file)], config.maps,
                                      config.label_file_dir, config.label_files)

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

        # Save predictions as SimpleITK images and save other images
        subject_results = os.path.join(result_dir, subject_name)
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

    roi_calculator = eval.ROICalculator(config.maps)
    summaries = evaluator.get_summaries()
    summaries.extend(roi_calculator.calculate(evaluator_roi.results, config.roi_reference_file))

    eval.SummaryResultWriter(summary_file).write(summaries)
    stat.QuantitativePlotter(result_dir).plot(csv_file, 'summary')
    stat.QuantitativeROIPlotter(result_dir, config.maps).plot(csv_roi_file, config.roi_reference_file, '')
    roi_calculator.save_summary(evaluator_roi.results, config.roi_reference_file, csv_roi_summary_file)


class MRFTensorFlowTester(tftest.TensorFlowTester):

    def __init__(self, data_handler: hdlr.DataHandler, model: tfmdl.TensorFlowModel, model_dir: str, result_dir: str,
                 config: cfg.Configuration, session: tf.Session):
        super().__init__(data_handler, model, model_dir, session)
        self.result_dir = result_dir
        self.config = config

    def init_subject_assembler(self) -> pymia_asmbl.SubjectAssembler:
        return asmbl.init_subject_assembler()

    def process_predictions(self, subject_assembler: pymia_asmbl.SubjectAssembler):
        process_predictions(self, subject_assembler, self.result_dir, self.config)

    def batch_to_feed_dict(self, batch: dict):
        feed_dict = {self.model.x_placeholder: np.stack(batch[pymia_def.KEY_IMAGES], axis=0),
                     self.model.is_training_placeholder: False}
        return feed_dict
