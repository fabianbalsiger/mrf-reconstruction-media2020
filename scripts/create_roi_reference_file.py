"""Creates a region of interest (ROI) reference file.

The ROI reference file is used during evaluation to assess the quality of the reconstruction.
"""
import argparse
import os

import numpy as np
import pymia.data.definition as pymia_def
import pymia.evaluation.evaluator as pymia_eval
import pymia.data.extraction as pymia_extr
import pymia.data.extraction.indexing as pymia_idx

import mrf.data.definition as defs
import mrf.data.extraction as ext
import mrf.utilities.evaluation as eval_
import mrf.utilities.normalization as norm


def main(hdf_file: str, out_file: str):
    maps = (defs.ID_MAP_FF, defs.ID_MAP_T1H2O, defs.ID_MAP_T1FAT, defs.ID_MAP_DF, defs.ID_MAP_B1)
    dataset = pymia_extr.ParameterizableDataset(hdf_file,
                                                pymia_idx.EmptyIndexing(),
                                                pymia_extr.ComposeExtractor(
                                                    [pymia_extr.SubjectExtractor(),
                                                     pymia_extr.NamesExtractor(),
                                                     pymia_extr.DataExtractor(
                                                         categories=(defs.ID_MASK_FG,
                                                                     defs.ID_MASK_T1H2O,
                                                                     defs.ID_MASK_ROI,
                                                                     defs.ID_MASK_ROI_T1H2O)),
                                                     pymia_extr.SelectiveDataExtractor(selection=maps,
                                                                                       category=pymia_def.KEY_LABELS),
                                                     ext.NormalizationExtractor()
                                                     ]))

    evaluator = eval_.ROIEvaluator([pymia_eval.CSVEvaluatorWriter(out_file)], maps,
                                   '../data/labels',
                                   ['labels_legs.txt', 'labels_thighs.txt'],
                                   dict(MEAN=np.mean, STD=np.std, MEDIAN=np.median, NOVOXELS=np.count_nonzero))

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    for subject in dataset:
        subject_name = subject[pymia_def.KEY_SUBJECT]
        print(subject_name)

        maps_reference = norm.process(subject[pymia_def.KEY_LABELS], subject[defs.ID_MASK_FG], subject[defs.KEY_NORM], maps)

        masks = {'FG': subject[defs.ID_MASK_FG], 'T1H2O': subject[defs.ID_MASK_FG]}
        roi_masks = {'FG': subject[defs.ID_MASK_ROI], 'T1H2O': subject[defs.ID_MASK_ROI_T1H2O]}
        evaluator.evaluate(maps_reference, roi_masks, masks, subject_name)

    evaluator.write()


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Region of interest (ROI) reference file creation')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../data/data.h5',
        help='Path to the dataset.'
    )

    parser.add_argument(
        '--out_file',
        type=str,
        default='../data/roi.csv',
        help='Path to the ROI reference file.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.out_file)
