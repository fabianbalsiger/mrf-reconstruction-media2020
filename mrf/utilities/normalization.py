import numpy as np


def process(map_arr: np.ndarray, mask_arr: np.ndarray, norm_dict: dict, maps: list) -> np.ndarray:
    mask_arr = np.squeeze(mask_arr)  # due to pymia dataset convention
    for map_idx, map_name in enumerate(maps):
        map_data = map_arr[..., map_idx]

        # rescale to original intensity range
        map_data = map_data * (norm_dict['max'][map_name] - norm_dict['min'][map_name]) + norm_dict['min'][map_name]
        # clip to original intensity range
        map_data[map_data > norm_dict['max'][map_name]] = norm_dict['max'][map_name]
        map_data[map_data < norm_dict['min'][map_name]] = norm_dict['min'][map_name]
        # mask background to zero
        map_data[mask_arr == 0] = 0

        map_arr[..., map_idx] = map_data
    return map_arr


def de_normalize(data: np.ndarray, norm_dict: dict, maps: list):
    for idx, map_name in enumerate(maps):
        data[..., idx] = data[..., idx] * (norm_dict['max'][map_name] - norm_dict['min'][map_name]) + norm_dict['min'][map_name]
    return data
