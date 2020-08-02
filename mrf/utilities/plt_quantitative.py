import math
import os
import typing

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import scipy.stats

import mrf.data.definition as defs


def get_map_description(map_: str, with_unit: bool = True):
    if map_ == defs.ID_MAP_T1H2O:
        return '$\mathrm{T1_{H2O}}$ (ms)' if with_unit else '$\mathrm{T1_{H2O}}$'
    if map_ == defs.ID_MAP_T1FAT:
        return '$\mathrm{T1_{fat}}$ (ms)' if with_unit else '$\mathrm{T1_{fat}}$'
    elif map_ == defs.ID_MAP_FF:
        return 'FF'
    elif map_ == defs.ID_MAP_DF:
        return '$\Delta$f (Hz)' if with_unit else '$\Delta$f'
    elif map_ == defs.ID_MAP_B1:
        return 'B1 (a.u.)' if with_unit else 'B1'
    else:
        raise ValueError('Map {} not supported'.format(map_.replace('map', '')))


def get_metric_long_description(metric: str, with_unit: bool = False):
    if metric == 'R2':
        return 'Coefficient of determination $\mathrm{R^2}$'
    if metric == 'MAE':
        return 'Mean absolute error MAE'
    if metric == 'MSE':
        return 'Mean squared error MSE'
    if metric == 'RMSE':
        return 'Root mean squared error RMSE'
    if metric == 'NRMSE':
        return 'Normalized root mean squared error NRMSE'
    if metric == 'PSNR':
        return 'Peak signal-to-noise ratio PSNR (dB)' if with_unit else 'Peak signal-to-noise ratio PSNR'
    if metric == 'SSIM':
        return 'Structural similarity index metric SSIM'
    else:
        return metric


class QuantitativePlotter:

    def __init__(self, plot_path: str, metrics: tuple = ('NRMSE', ), plot_format: str = 'png'):
        self.path = plot_path
        self.metrics = metrics
        self.plot_format = plot_format

        os.makedirs(self.path, exist_ok=True)

    def plot(self, csv_file: str, file_name: str = 'summary', plot_images: bool = True):
        df = pd.read_csv(csv_file, sep=';')
        experiment, _ = os.path.splitext(os.path.basename(csv_file))

        plotly_figs = []
        plotly_titles = []
        plotly_yaxis = []
        plotly_minmax = []

        for map_ in df['MAP'].unique():
            for mask in df['MASK'].unique():

                values = df[(df['MAP'] == map_) & (df['MASK'] == mask)]

                # not all map mask combinations have been calculated on CSV file generation
                if values.count().any() == 0:
                    continue

                for metric in self.metrics:
                    data = values[metric].values

                    if plot_images:
                        self.plot_box(
                            os.path.join(self.path, '{}_{}_{}.{}'.format(map_, metric, mask, self.plot_format)),
                            data,
                            '{} ({})\n{}'.format(map_, mask, experiment),
                            '',
                            get_metric_long_description(metric),
                            self._get_min_of_metric(metric), self._get_max_of_metric(metric)
                        )

                    plotly_figs.append(go.Box(y=values[metric],
                                              text=values['ID'],
                                              boxpoints='outliers',
                                              marker=dict(color='rgb(0, 0, 0)'),
                                              fillcolor='rgba(255,255,255,0)',
                                              boxmean='sd',
                                              name=''
                                              ))
                    plotly_titles.append(map_)
                    plotly_yaxis.append(get_metric_long_description(metric))
                    plotly_minmax.append((self._get_min_of_metric(metric) if self._get_min_of_metric(metric) is not None else min(values[metric]) * 0.9,
                                          self._get_max_of_metric(metric) if self._get_max_of_metric(metric) is not None else max(values[metric]) * 1.1))

        cols = len(self.metrics)
        rows = math.ceil(len(plotly_figs) / cols)
        fig = tls.make_subplots(rows=rows, cols=cols, subplot_titles=plotly_titles, print_grid=False)

        fig_idx = 0
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                if fig_idx < len(plotly_figs):
                    fig.append_trace(plotly_figs[fig_idx], row, col)
                    fig_idx += 1

        fig['layout'].update(title=experiment, showlegend=False, height=rows*400)
        for idx in range(1, len(plotly_figs) + 1):
            fig['layout']['yaxis{}'.format(idx)].update(range=plotly_minmax[idx - 1], title=plotly_yaxis[idx - 1])
            fig['layout']['xaxis{}'.format(idx)].update(title='')

        py.offline.plot(fig, filename=os.path.join(self.path, file_name + '.html'), auto_open=False)

    @staticmethod
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    @staticmethod
    def set_box_format(bp):
        plt.setp(bp['caps'], linewidth=0)
        plt.setp(bp['medians'], linewidth=1.5)
        plt.setp(bp['fliers'], marker='.')
        plt.setp(bp['fliers'], markerfacecolor='black')
        plt.setp(bp['fliers'], alpha=1)

    def plot_box(self, file_path: str, data, title: str, x_label: str, y_label: str,
                 min_: float = None, max_: float = None):
        fig = plt.figure(figsize=plt.rcParams["figure.figsize"][::-1])  # figsize defaults to (width, height)=(6.4, 4.8)
        # for boxplots, we want the ratio to be inversed
        ax = fig.add_subplot(111)  # create an axes instance (nrows=ncols=index)
        bp = ax.boxplot(data, widths=0.6)
        self.set_box_format(bp)

        ax.set_title(title)
        ax.set_ylabel(y_label)
        if x_label is not None:
            ax.set_xlabel(x_label)

        # remove frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # thicken frame
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        # adjust min and max if provided
        if min_ is not None or max_ is not None:
            min_original, max_original = ax.get_ylim()
            min_ = min_ if min_ is not None and min_ < min_original else min_original
            max_ = max_ if max_ is not None and max_ > max_original else max_original
            ax.set_ylim(min_, max_)

        plt.savefig(file_path)
        plt.close()

    @staticmethod
    def _get_min_of_metric(metric: str):
        if metric == 'R2':
            return 0
        if metric == 'MAE':
            return 0
        if metric == 'MSE':
            return 0
        if metric == 'RMSE':
            return 0
        if metric == 'NRMSE':
            return 0
        if metric == 'PSNR':
            return 0
        if metric == 'SSIM':
            return 0
        else:
            return None  # we do not raise an error

    @staticmethod
    def _get_max_of_metric(metric: str):
        if metric == 'R2':
            return 1
        if metric == 'MAE':
            return None
        if metric == 'MSE':
            return None
        if metric == 'RMSE':
            return None
        if metric == 'NRMSE':
            return None
        if metric == 'PSNR':
            return None
        if metric == 'SSIM':
            return 1
        else:
            return None  # we do not raise an error


class QuantitativeROIPlotter:

    def __init__(self, plot_path: str, maps: tuple = (defs.ID_MAP_T1H2O, defs.ID_MAP_T1FAT, defs.ID_MAP_FF,
                                                      defs.ID_MAP_DF, defs.ID_MAP_B1),
                 plot_format: str = 'png'):
        self.path = plot_path
        self.format = plot_format
        self.maps = maps

        os.makedirs(self.path, exist_ok=True)

    def plot(self, prediction_csv_file: str, reference_csv_file: str, suffix: str = 'summary'):
        metric = 'MEAN'
        prediction = pd.read_csv(prediction_csv_file, sep=';')
        reference = pd.read_csv(reference_csv_file, sep=';')

        df = pd.merge(prediction, reference, how='left', on=['ID', 'MASK', 'MAP', 'ROI', 'SLICE'], suffixes=('', '_REF'))

        for map_ in self.maps:
            map_short = map_.replace('map', '')
            values = df[(df['MAP'] == map_short)]
            values = values.dropna()  # due to masking of T1H2O, there might be NaNs
            # not all map mask combinations have been calculated on CSV file generation
            if values.count().any() == 0:
                continue

            unit = get_map_description(map_)
            values_predicted = values[metric].values
            values_reference = values[metric + '_REF'].values

            if suffix:
                file_name_scaffold = os.path.join(self.path, '{}_{}_{}.{}'.format(map_short, '{}', suffix, self.format))
            else:
                file_name_scaffold = os.path.join(self.path, '{}_{}.{}'.format(map_short, '{}', self.format))

            with plt.rc_context({'font.weight': 'bold', 'font.size': 12, 'mathtext.default': 'regular'}):
                self.correlation_plot(file_name_scaffold.format('SCATTER'),
                                      values_reference, values_predicted,
                                      'Reference {}'.format(unit), 'Predicted {}'.format(unit), '')

                self.residual_plot(file_name_scaffold.format('RESIDUAL'),
                                   values_predicted, values_reference,
                                   'Predicted {}'.format(unit), 'Residual {}'.format(unit), '')

                self.bland_altman_plot(file_name_scaffold.format('BLANDALTMAN'),
                                       values_predicted, values_reference, unit)

    @staticmethod
    def bland_altman_plot(path, data1: np.ndarray, data2: np.ndarray, variable_name):
        mean = np.mean([data1, data2], axis=0)
        diff = data1 - data2
        md = np.mean(diff)  # mean of the difference
        sd = np.std(diff, axis=0)  # standard deviation of the difference

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)  # create an axes instance (nrows=ncols=index)
        ax.scatter(mean, diff, s=2, color='black')

        # ax.set_title('Bland-Altman')
        ax.set_ylabel('$\Delta${}'.format(variable_name), fontweight='bold', fontsize=12)
        ax.set_xlabel(variable_name, fontweight='bold', fontsize=12)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.axhline(md, color='gray', linestyle='-')
        plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
        plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')

        # https://stackoverflow.com/questions/43675355/python-text-to-second-y-axis?noredirect=1&lq=1
        _, x = plt.gca().get_xlim()
        plt.text(x, md + 1.96 * sd, '+1.96 SD', ha='left', va='bottom')
        plt.text(x, md + 1.96 * sd, '{:.3f}'.format(md + 1.96 * sd), ha='left', va='top')

        plt.text(x, md - 1.96 * sd, '-1.96 SD', ha='left', va='bottom')
        plt.text(x, md - 1.96 * sd, '{:.3f}'.format(md - 1.96 * sd), ha='left', va='top')

        plt.text(x, md, 'Mean', ha='left', va='bottom')
        plt.text(x, md, '{:.3f}'.format(md), ha='left', va='top')

        fig.subplots_adjust(right=0.89)  # adjust slightly such that "+1.96 SD" is not cut off

        plt.savefig(path)
        plt.close()

    @staticmethod
    def residual_plot(path, predicted, reference, x_label, y_label, title):

        # Create the plot object
        _, ax = plt.subplots(figsize=(8, 6))

        # Plot the data, set the size (s), color and transparency (alpha)
        # of the points
        residuals = reference - predicted

        ax.scatter(predicted, residuals, s=2, color='black')

        # Label the axes and provide a title
        ax.set_title(title)
        ax.set_xlabel(x_label, fontweight='bold', fontsize=12)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=12)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.savefig(path)
        plt.close()

    def correlation_plot(self, path, x_data, y_data, x_label, y_label, title,
                         with_regression_line: bool = True, with_confidence_interval: bool = True,
                         with_abline: bool = True):

        # Create the plot object
        _, ax = plt.subplots(figsize=(8, 6))

        # Plot the data, set the size (s), color and transparency (alpha)
        # of the points
        ax.scatter(x_data, y_data, s=2, color='black')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # add abline
        if with_abline:
            x = np.linspace(*ax.get_xlim())
            ax.plot(x, x, color='gray')

        z = np.polyfit(x_data, y_data, 1)

        if with_regression_line:
            p = np.poly1d(z)
            fit = p(x_data)

            # get the coordinates for the fit curve
            c_y = [np.min(fit), np.max(fit)]
            c_x = [np.min(x_data), np.max(x_data)]

            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_data, y_data)
            regression = 'y = {:.3f}x {} {:.3f}'.format(slope, '+' if intercept > 0 else '-', abs(intercept))
            correlation = 'r = {:.3f}, {}'.format(r_value, self.get_p_value(p_value))

            # plot line of best fit
            ax.plot(c_x, c_y, color='gray', linestyle='dashed', label=regression)
            ax.plot([], [], ' ', label=correlation)  # "plot" to show r in legend

            # for legend placement see: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
            ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.1))

        # Label the axes and provide a title
        ax.set_title(title)
        ax.set_xlabel(x_label, fontweight='bold', fontsize=12)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=12)

        plt.savefig(path)
        plt.close()

    @staticmethod
    def get_p_value(p_value: float):
        if p_value < 0.001:
            return 'p < 0.001' # ***
        else:
            return 'p = {:.3f}'.format(p_value)
