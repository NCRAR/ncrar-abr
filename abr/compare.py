import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from atom.api import Bool, Dict, Event, List, observe, Typed, Str
import enaml
from enaml.core.api import d_, Declarative
from enaml.qt.qt_application import QtApplication

with enaml.imports():
    from abr.main_window import CompareWindow

from abr.app import add_default_arguments, parse_args


class Compare(Declarative):

    waves = Typed(pd.DataFrame)

    as_difference = d_(Bool(False))
    jitter = d_(Bool(False))
    axes = Typed(plt.Axes)
    figure = Typed(plt.Figure)
    plot_map = Dict()

    rater_x = d_(Str())
    rater_y = d_(Str())
    selected_feature = d_(Str())
    selected_points = d_(List())
    selected_points_updated = d_(Event(), writable=False)

    available_raters = List()
    available_features = List()
    available_subjects = List()


    def _observe_waves(self, event):
        features = [c for c in self.waves.columns if 'msec' not in c.lower()]
        features.sort(key=lambda x: (int(x[1]), (x[0] != 'P')))
        self.available_features = features
        self.available_raters = self.waves.index.unique('analyzer').tolist()

    def _default_figure(self):
        context = {
            'axes.spines.left': True,
            'ytick.left': True,
            'figure.subplot.left': 0.15,
            'figure.subplot.bottom': 0.1,
            'figure.subplot.top': 0.95,
            'figure.subplot.right': 0.95,
        }
        with plt.rc_context(context):
            figure, self.axes = plt.subplots(1, 1)
        return figure

    def _default_rater_x(self):
        return self.available_raters[0]

    def _default_rater_y(self):
        i = 1 if (len(self.available_raters) > 1) else 0
        return self.available_raters[i]

    def _default_selected_feature(self):
        return self.available_features[0]

    @observe('rater_x', 'rater_y', 'as_difference', 'jitter', 'selected_feature')
    def _update_plot(self, event=None):
        self.axes.clear()
        self.plot_map = {}
        data = self.waves[self.selected_feature].unstack('analyzer')
        x = data[self.rater_x].copy()
        y = data[self.rater_y].copy()

        self.axes.set_xlabel(f'Rater {self.rater_x}')
        if self.as_difference:
            y -= x
            self.axes.set_ylabel(f'Difference between {self.rater_y} and {self.rater_x}')
        else:
            self.axes.set_ylabel(f'Rater {self.rater_y}')


        if self.jitter:
            bound = (x.max() - x.min()) * 0.025
            x += np.random.uniform(-bound, bound, len(x))
            y += np.random.uniform(-bound, bound, len(x))

        print(x)
        for (filename, frequency), _ in data.groupby(['filename', 'frequency']):
            xd = x.xs(filename, level='filename').xs(frequency, level='frequency')
            yd = y.xs(filename, level='filename').xs(frequency, level='frequency')
            l, = self.axes.plot(xd, yd, 'o', picker=4, mec='w', mew=1)
            self.plot_map[l] = filename, frequency

        if self.figure.canvas is not None:
            self.figure.canvas.draw()

    def pick_handler(self, event):
        filename, frequency = self.plot_map[event.artist]
        self.selected_points = [(filename, frequency * 1e3)]
        self.selected_points_updated = True


def main():
    parser = argparse.ArgumentParser("abr_compare")
    add_default_arguments(parser, waves=False)
    parser.add_argument('directory')
    options = parse_args(parser, waves=False)

    app = QtApplication()
    _, waves = options['parser'].load_analyses(options['directory'])
    waves = waves \
        .set_index(['filename', 'subject', 'frequency', 'Level', 'Replicate', 'analyzer']) \
        .sort_index()

    compare = Compare(waves=waves)
    view = CompareWindow(parser=options['parser'], compare=compare)
    view.show()
    app.start()
    app.stop()
