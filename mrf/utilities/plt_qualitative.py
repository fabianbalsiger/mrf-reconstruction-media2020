import os
import typing

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import mrf.data.definition as defs
import mrf.utilities.plt_quantitative as pltq


def plot_2d_image_colorbar(path: str, image: np.ndarray, cmap=cm.get_cmap('Greys'),
                           min_value: float=None, max_value: float=None, plot_image_size: int = 10,
                           legend: str = None):
    """Plots a 2-D image with a color bar to the right.

    See Also:
        Matplotlib colormaps: https://matplotlib.org/examples/color/colormaps_reference.html.
        Please use a perceptually uniform colormap like cm.viridis, cm.plasma, cm.inferno, or cm.magma.

    Args:
        path (str): The full path to save the plot.
        image (np.ndarray): The 2-D image.
        cmap (colormap): The color map.
        min_value (float): The min value of the color map. If None, then use the data's min value.
        max_value (float): The max value of the color map. If None, then use the data's max value.
        plot_image_size (int): The size of the image in inches. If the image is not rectangular, the longer size will
            be of dimension plot_image_size. Note that the width of the final figure will be larger than plot_image_size
            due to the colorbar.

    Returns:
        None.
    """

    # calculate figure size, unit is inches

    y, x = image.shape
    if x > y:
        y = y / x * plot_image_size
        x = plot_image_size
    elif y > x:
        x = x / y * plot_image_size
        y = plot_image_size
    else:
        x = y = plot_image_size
    width_of_colorbar = 5  # we need two inches of space for the colorbar (5 for large)
    x = x + width_of_colorbar
    colorbar_begin = 1 - width_of_colorbar / x

    fig = plt.figure(figsize=(x, y))  # add 1 to width for colorbar
    ax = fig.add_axes([0, 0, colorbar_begin, 1])
    ax.set_axis_off()
    ax.margins(0)
    img = ax.imshow(image, interpolation='nearest', cmap=cmap)
    cax = fig.add_axes([colorbar_begin + 0.01, 0.05, width_of_colorbar / x / 3, 0.9])
    cb = fig.colorbar(img, cax=cax)

    # adjust min and max if provided
    if min_value or max_value:
        min_, max_ = img.get_clim()
        min_ = min_value if min_value else min_
        max_ = max_value if max_value else max_
        img.set_clim(min_, max_)
        cb.set_clim(min_, max_value)
    cb.ax.tick_params(labelsize=50, width=4, size=20)
    # for better readability of colorbar labels
    plt.yticks(fontweight='bold')

    if legend:
        # plot legend to colorbar
        cb.ax.set_ylabel(legend, fontsize=20, weight='bold')

    plt.savefig(path, transparent=True)
    plt.close()


def get_colormap(map_: str):
    if map_ == defs.ID_MAP_T1H2O:
        return cm.get_cmap('viridis')
    elif map_ == defs.ID_MAP_T1FAT:
        return cm.get_cmap('gray')
    elif map_ == defs.ID_MAP_FF:
        return cm.get_cmap('gray')
    elif map_ == defs.ID_MAP_B1:
        return cm.get_cmap('plasma')
    elif map_ == defs.ID_MAP_DF:
        return cm.get_cmap('inferno')
    else:
        raise ValueError('Map {} not supported'.format(map_.replace('map', '')))


class QualitativePlotter:

    def __init__(self, path: str, slice_no: int = 2, plot_format: str = 'png', with_legend: bool = False):
        self.path = path
        self.slice_no = slice_no
        self.plot_format = plot_format
        self.with_legend = with_legend

        os.makedirs(self.path, exist_ok=True)

    def plot(self, subject: str, map_name: str, prediction: np.ndarray, ground_truth: np.ndarray, mask_fg: np.ndarray,
             min_error: float = None, max_error: float = None):
        if self.with_legend:
            legend = pltq.get_map_description(map_name)
        else:
            legend = None

        mask = np.squeeze(mask_fg)  # due to dataset convention
        map_name_short = map_name.replace('map', '')

        # calculate error
        error = prediction - ground_truth
        error[mask == 0] = 0  # the min intensity of the ground truth might not be 0,
        # i.e. background would have an error

        path = os.path.join(self.path, '{}_{}_{}_{}.{}'.format(subject,
                                                               map_name_short,
                                                               '{}',
                                                               self.slice_no,
                                                               self.plot_format))

        plot_2d_image_colorbar(path.format('PRED'), prediction[self.slice_no, ...],
                               min_value=ground_truth.min(), max_value=ground_truth.max(),
                               cmap=get_colormap(map_name), legend=legend)

        plot_2d_image_colorbar(path.format('ERR'), error[self.slice_no, ...], cmap=cm.get_cmap('cividis'),
                               legend=legend, min_value=min_error, max_value=max_error)
