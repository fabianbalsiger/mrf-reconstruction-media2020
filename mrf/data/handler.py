import pymia.data.definition as pymia_def
import pymia.data.extraction as pymia_extr
import pymia.data.transformation as pymia_tfm
import pymia.deeplearning.data_handler as hdlr
import pymia.deeplearning.tensorflow.conversion as lib_cnv

import mrf.configuration.config as cfg
import mrf.data.definition as defs
import mrf.data.extraction as ext
import mrf.data.pickler as pkl


class MRFDataHandler(hdlr.DataHandler):

    def __init__(self, config: cfg.Configuration,
                 subjects_train,
                 subjects_valid,
                 subjects_test,
                 is_subject_selection: bool = True,
                 collate_fn=lib_cnv.TensorFlowCollate(),
                 padding_size: tuple = (0, 0, 0)):
        super().__init__()

        indexing_strategy = pymia_extr.PatchWiseIndexing(patch_shape=config.patch_size, ignore_incomplete=False)

        self.dataset = pymia_extr.ParameterizableDataset(config.database_file,
                                                         indexing_strategy,
                                                         pymia_extr.SubjectExtractor(),  # for the select_indices
                                                         None)

        self.no_subjects_train = len(subjects_train)
        self.no_subjects_valid = len(subjects_valid)
        self.no_subjects_test = len(subjects_test)

        if is_subject_selection:
            # get sampler ids by subjects
            sampler_ids_train = pymia_extr.select_indices(self.dataset, pymia_extr.SubjectSelection(subjects_train))
            sampler_ids_valid = pymia_extr.select_indices(self.dataset, pymia_extr.SubjectSelection(subjects_valid))
            sampler_ids_test = pymia_extr.select_indices(self.dataset, pymia_extr.SubjectSelection(subjects_test))
        else:
            # get sampler ids from indices files
            sampler_ids_train, sampler_ids_valid, sampler_ids_test = pkl.load_sampler_ids(
                config.indices_dir,
                pkl.PATCH_WISE_FILE_NAME,
                subjects_train, subjects_valid, subjects_test)

        # define extractors
        self.extractor_train = pymia_extr.ComposeExtractor(
            [pymia_extr.NamesExtractor(),  # required for SelectiveDataExtractor
             pymia_extr.PadDataExtractor(padding=padding_size,
                                         extractor=pymia_extr.DataExtractor(categories=(pymia_def.KEY_IMAGES,))),
             pymia_extr.PadDataExtractor(padding=(0, 0, 0),
                                         extractor=pymia_extr.SelectiveDataExtractor(selection=config.maps,
                                                                                     category=pymia_def.KEY_LABELS)),
             pymia_extr.PadDataExtractor(padding=(0, 0, 0),
                                         extractor=pymia_extr.DataExtractor(
                                             categories=(defs.ID_MASK_FG, defs.ID_MASK_T1H2O))),
             pymia_extr.IndexingExtractor(),
             pymia_extr.ImageShapeExtractor()
             ])

        # to calculate validation loss, we require the labels and mask during validation
        self.extractor_valid = pymia_extr.ComposeExtractor(
            [pymia_extr.NamesExtractor(),  # required for SelectiveDataExtractor
             pymia_extr.PadDataExtractor(padding=padding_size,
                                         extractor=pymia_extr.DataExtractor(categories=(pymia_def.KEY_IMAGES,))),
             pymia_extr.PadDataExtractor(padding=(0, 0, 0),
                                         extractor=pymia_extr.SelectiveDataExtractor(selection=config.maps,
                                                                                     category=pymia_def.KEY_LABELS)),
             pymia_extr.PadDataExtractor(padding=(0, 0, 0),
                                         extractor=pymia_extr.DataExtractor(
                                             categories=(defs.ID_MASK_FG, defs.ID_MASK_T1H2O,
                                                         defs.ID_MASK_ROI, defs.ID_MASK_ROI_T1H2O))),
             pymia_extr.IndexingExtractor(),
             pymia_extr.ImageShapeExtractor()
             ])

        self.extractor_test = pymia_extr.ComposeExtractor(
            [pymia_extr.NamesExtractor(),  # required for SelectiveDataExtractor
             pymia_extr.SubjectExtractor(),
             pymia_extr.SelectiveDataExtractor(selection=config.maps, category=pymia_def.KEY_LABELS),
             pymia_extr.DataExtractor(categories=(defs.ID_MASK_FG, defs.ID_MASK_T1H2O,
                                                  defs.ID_MASK_ROI, defs.ID_MASK_ROI_T1H2O)),
             pymia_extr.ImagePropertiesExtractor(),
             pymia_extr.ImageShapeExtractor(),
             ext.NormalizationExtractor()
             ])

        # define transforms for extraction
        # after extraction, the first dimension is the batch dimension.
        # E.g., shape = (1, 16, 16, 4) instead of (16, 16, 4) --> therefore squeeze the data
        self.extraction_transform_train = pymia_tfm.Squeeze(
            entries=(pymia_def.KEY_IMAGES, pymia_def.KEY_LABELS, defs.ID_MASK_FG, defs.ID_MASK_T1H2O), squeeze_axis=0)
        self.extraction_transform_valid = pymia_tfm.Squeeze(
            entries=(pymia_def.KEY_IMAGES, pymia_def.KEY_LABELS, defs.ID_MASK_FG, defs.ID_MASK_T1H2O), squeeze_axis=0)
        self.extraction_transform_test = None

        # define loaders
        training_sampler = pymia_extr.SubsetRandomSampler(sampler_ids_train)
        self.loader_train = pymia_extr.DataLoader(self.dataset,
                                                  config.batch_size_training,
                                                  sampler=training_sampler,
                                                  collate_fn=collate_fn,
                                                  num_workers=1)

        validation_sampler = pymia_extr.SubsetSequentialSampler(sampler_ids_valid)
        self.loader_valid = pymia_extr.DataLoader(self.dataset,
                                                  config.batch_size_testing,
                                                  sampler=validation_sampler,
                                                  collate_fn=collate_fn,
                                                  num_workers=1)

        testing_sampler = pymia_extr.SubsetSequentialSampler(sampler_ids_test)
        self.loader_test = pymia_extr.DataLoader(self.dataset,
                                                 config.batch_size_testing,
                                                 sampler=testing_sampler,
                                                 collate_fn=collate_fn,
                                                 num_workers=1)
