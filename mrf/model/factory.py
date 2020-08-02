import mrf.configuration.config as cfg

import mrf.model.media as media


MODEL_UNKNOWN_ERROR_MESSAGE = 'Unknown model "{}".'


def get_information(config: cfg.Configuration):
    if config.model == media.MODEL_MEDIA:
        padding = (config.receptive_field - 1) // 2
        return media.MEDIAModel, (0, padding, padding)
    else:
        raise ValueError(MODEL_UNKNOWN_ERROR_MESSAGE.format(config.model))


def get_model(config: cfg.Configuration):
    return get_information(config)[0]


def get_padding_size(config: cfg.Configuration):
    return get_information(config)[1]
