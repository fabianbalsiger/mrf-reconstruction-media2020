import pymia.config.configuration as cfg
import pymia.deeplearning.config as dlcfg

import mrf.data.definition as defs


class Configuration(dlcfg.DeepLearningConfiguration):
    """Represents a configuration."""

    VERSION = 1
    TYPE = 'MAIN'

    @classmethod
    def version(cls) -> int:
        return cls.VERSION

    @classmethod
    def type(cls) -> str:
        return cls.TYPE

    def __init__(self):
        """Initializes a new instance of the Configuration class."""
        super().__init__()
        self.indices_dir = ''
        self.split_file = ''
        self.label_file_dir = ''
        self.label_files = ['labels_legs.txt', 'labels_thighs.txt']
        self.roi_reference_file = ''

        self.experiment = ''  # string to describe experiment

        # model configuration
        self.model = ''  # string identifying the model
        self.maps = [defs.ID_MAP_FF, defs.ID_MAP_T1H2O, defs.ID_MAP_T1FAT, defs.ID_MAP_DF, defs.ID_MAP_B1]
        self.patch_size = [1, 32, 32]
        self.receptive_field = 15
        self.no_parameters = 5000000
        self.no_nonlinearities = 21
        self.no_channels_last_spatial_block = 64
        self.no_channels_minimum_temporal_block = 32
        self.no_channels_decrease_spatial = 32
        self.no_channels_decrease_temporal = 32

        # training configuration
        self.learning_rate = 0.001  # the learning rate

        # we use the R2 as best model score
        self.best_model_score_is_positive = True
        self.best_model_score_name = 'r2'


def load(path: str, config_cls):
    """Loads a configuration file.

    Args:
        path (str): The path to the configuration file.
        config_cls (class): The configuration class (not an instance).

    Returns:
        (config_cls): The configuration.
    """

    return cfg.load(path, config_cls)
