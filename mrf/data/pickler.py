import os
import pickle


PATCH_WISE_FILE_NAME = 'patchwise-{}.pkl'


def dump_indices_file(file: str, file_name: str, ids: list):
    with open(os.path.join(file, file_name), 'wb') as f:
        pickle.dump(ids, f)


def load_indices_file(file: str) -> list:
    with open(file, 'rb') as f:
        indices = pickle.load(f)
    return indices


def load_sampler_ids(indices_dir: str, file_name: str, subjects_train, subjects_valid, subject_test):
    sampler_ids_train = []
    for subject in subjects_train:
        sampler_ids_train.extend(load_indices_file(
            os.path.join(indices_dir, file_name.format(subject))))

    sampler_ids_valid = []
    for subject in subjects_valid:
        sampler_ids_valid.extend(load_indices_file(
            os.path.join(indices_dir, file_name.format(subject))))

    sampler_ids_test = []
    for subject in subject_test:
        sampler_ids_test.extend(load_indices_file(
            os.path.join(indices_dir, file_name.format(subject))))

    return sampler_ids_train, sampler_ids_valid, sampler_ids_test
