"""Creates the indices files for each subject, which are used to do the patch extraction."""
import argparse
import os

import pymia.data.extraction as pymia_extr
import pymia.data.extraction.indexing as pymia_idx

import mrf.data.definition as defs
import mrf.data.pickler as pkl


class WithForegroundSelectionByMask(pymia_extr.SelectionStrategy):

    def __init__(self, mask: str):
        self.mask = mask

    def __call__(self, sample) -> bool:
        return (sample[self.mask]).any()


def main(hdf_file: str, save_dir: str):
    mask = defs.ID_MASK_FG  # we only want to train on patches that contain foreground, and, therefore use the foreground mask
    patch_shape = (1, 32, 32)  # this corresponds to the output patch size

    indexing_strategy = pymia_idx.PatchWiseIndexing(patch_shape=patch_shape, ignore_incomplete=False)
    file_str = pkl.PATCH_WISE_FILE_NAME

    dataset = pymia_extr.ParameterizableDataset(hdf_file,
                                                indexing_strategy,
                                                pymia_extr.ComposeExtractor(
                                                    [pymia_extr.SubjectExtractor(),
                                                     pymia_extr.DataExtractor(categories=(mask, ))]))

    os.makedirs(save_dir, exist_ok=True)

    for subject in dataset.get_subjects():
        print(subject)

        selection = pymia_extr.ComposeSelection([pymia_extr.SubjectSelection(subject),
                                                 WithForegroundSelectionByMask(mask)])

        ids = pymia_extr.select_indices(dataset, selection)
        pkl.dump_indices_file(save_dir, file_str.format(subject), ids)


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Dataset indices file creation')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../data/data.h5',
        help='Path to the dataset.'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='../data/indices',
        help='Path to the indices file directory.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.save_dir)
