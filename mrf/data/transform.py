import numpy as np
import pymia.data.definition as pymia_def
import pymia.data.transformation as pymia_tfm


class ParametricMapNormalization(pymia_tfm.Transform):
    """Normalizes the parametric maps to [0, 1]."""

    def __init__(self, min_: dict, max_: dict) -> None:
        super().__init__()
        self.min_ = min_
        self.max_ = max_

    def __call__(self, sample: dict) -> dict:
        maps = pymia_tfm.check_and_return(sample[pymia_def.KEY_LABELS], np.ndarray)

        for idx, entry in enumerate(self.min_.keys()):
            maps[..., idx] = (maps[..., idx] - self.min_[entry]) / (self.max_[entry] - self.min_[entry])  # normalize to 0..1

        sample[pymia_def.KEY_LABELS] = maps
        return sample


class MaskedParametricMapNormalization(pymia_tfm.Transform):
    """Normalizes the parametric maps [0, 1] and sets the background to zero."""

    def __init__(self, min_: dict, max_: dict, mask: str) -> None:
        super().__init__()
        self.min_ = min_
        self.max_ = max_
        self.mask = mask

    def __call__(self, sample: dict) -> dict:
        # normalize first
        norm = ParametricMapNormalization(self.min_, self.max_)
        sample = norm(sample)

        maps = pymia_tfm.check_and_return(sample[pymia_def.KEY_LABELS], np.ndarray)
        mask = pymia_tfm.check_and_return(sample[self.mask], np.ndarray)

        mask = np.concatenate([mask] * maps.shape[-1], -1)
        maps[mask == 0] = 0

        sample[pymia_def.KEY_LABELS] = maps
        return sample


class TemporalPermutation(pymia_tfm.Transform):
    """Transformation for temporal permutation experiment."""

    def __init__(self, temporal_idx: int=0, entries=('images', )):
        super().__init__()
        self.temporal_idx = temporal_idx
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                raise ValueError(pymia_tfm.ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            np_entry = pymia_tfm.check_and_return(sample[entry], np.ndarray)
            frame = np_entry[..., self.temporal_idx, :]
            if len(frame.shape) == 3:  # without batch dimension
                shape = frame[..., 0].shape  # spatial dimension
                # permute real and imaginary independently
                frame[..., 0] = np.random.permutation(frame[..., 0].reshape(-1)).reshape(shape)
                frame[..., 1] = np.random.permutation(frame[..., 1].reshape(-1)).reshape(shape)
            else:
                # we have the batch dimension
                shape = frame[0, ..., 0].shape  # spatial dimension
                for batch_idx in range(frame.shape[0]):
                    # permute real and imaginary independently
                    frame[batch_idx, ..., 0] = np.random.permutation(frame[batch_idx, ..., 0].reshape(-1)).reshape(shape)
                    frame[batch_idx, ..., 1] = np.random.permutation(frame[batch_idx, ..., 1].reshape(-1)).reshape(shape)

            np_entry[..., self.temporal_idx, :] = frame
            sample[entry] = np_entry

        return sample
