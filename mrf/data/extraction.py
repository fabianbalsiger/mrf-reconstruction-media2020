import pymia.data.extraction as pymia_ext

import mrf.data.definition as defs


class NormalizationExtractor(pymia_ext.Extractor):
    """Extracts the normalization values from the dataset."""

    def extract(self, reader: pymia_ext.Reader, params: dict, extracted: dict) -> None:
        label_names = reader.read('meta/names/labels_names')
        mins = {}
        maxs = {}
        for label in label_names:
            mins[label] = reader.read('norm/min/{}'.format(label))
            maxs[label] = reader.read('norm/max/{}'.format(label))

        extracted[defs.KEY_NORM] = {'min': mins, 'max': maxs}
