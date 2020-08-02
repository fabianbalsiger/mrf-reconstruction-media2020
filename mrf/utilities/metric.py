import pymia.evaluation.metric as pymia_metric

import numpy as np
import SimpleITK as sitk
import skimage.metrics


def get_metrics():
    return [pymia_metric.MeanAbsoluteError(),
            pymia_metric.MeanSquaredError(), pymia_metric.RootMeanSquaredError(),
            pymia_metric.NormalizedRootMeanSquaredError(),
            PeakSignalToNoiseRatio(), StructuralSimilarityIndexMeasure()]


class PeakSignalToNoiseRatio(pymia_metric.INumpyArrayMetric):

    def __init__(self):
        super().__init__()
        self.metric = 'PSNR'

    def calculate(self):
        ground_truth = np.extract(self.ground_truth != 0, self.ground_truth)
        segmentation = np.extract(self.ground_truth != 0, self.segmentation)
        try:
            psnr = skimage.metrics.peak_signal_noise_ratio(ground_truth, segmentation, data_range=ground_truth.max())
        except ValueError:
            print(ground_truth)
            print(segmentation)
            return -1
        return psnr


class StructuralSimilarityIndexMeasure(pymia_metric.ISimpleITKImageMetric):

    def __init__(self):
        super().__init__()
        self.metric = 'SSIM'

    def calculate(self):
        ground_truth = sitk.GetArrayFromImage(self.ground_truth)
        ground_truth = ground_truth.transpose((2, 1, 0))
        segmentation = sitk.GetArrayFromImage(self.segmentation)
        segmentation = segmentation.transpose((2, 1, 0))
        ssim = skimage.metrics.structural_similarity(ground_truth, segmentation, data_range=ground_truth.max(),
                                                     multichannel=True)
        return ssim
