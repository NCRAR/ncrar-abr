import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from atom.api import Bool, Dict, Event, List, observe, Typed, Str
import enaml
from enaml.application import deferred_call
from enaml.core.api import d_, Declarative
from enaml.qt.qt_application import QtApplication

with enaml.imports():
    from abr.compare_window import CompareWindow

from abr.app import add_default_arguments, parse_args
from abr.presenter import WaveformPresenter


class Compare(Declarative):

    waves = Typed(pd.DataFrame)
    data = Typed(pd.DataFrame)

    as_difference = d_(Bool(False))
    jitter = d_(Bool(False))
    axes = Typed(plt.Axes)
    figure = Typed(plt.Figure)
    plot_map = Dict()

    rater_x = d_(Str())
    rater_y = d_(Str())
    selected_feature = d_(Str())
    selected_points = d_(List())

    available_raters = List()
    available_features = List()
    available_subjects = List()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_update()

    def _observe_waves(self, event):
        features = [c for c in self.waves.columns if 'msec' not in c.lower()]
        features.sort(key=lambda x: (int(x[1]), (x[0] != 'P')))
        self.available_features = features
        self.available_raters = self.waves.index.unique('analyzer').tolist()
        self.data = self.waves.reset_index('analyzed_filename').unstack('analyzer')

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
    def request_update(self, event=None):
        if event is not None and event['type'] == 'create':
            return
        deferred_call(self._update_plot)

    def _update_plot(self):
        self.axes.clear()
        self.plot_map = {}

        x = self.data.loc[:, (self.selected_feature, self.rater_x)].copy()
        y = self.data.loc[:, (self.selected_feature, self.rater_y)].copy()

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

        for (filename, frequency), _ in x.groupby(['filename', 'frequency']):
            xd = x.xs(filename, level='filename').xs(frequency, level='frequency')
            yd = y.xs(filename, level='filename').xs(frequency, level='frequency')
            l, = self.axes.plot(xd, yd, 'o', mec='w', mew=1)
            self.plot_map[l] = filename, frequency

        if self.figure.canvas is not None:
            self.figure.canvas.draw()

    def button_press_event(self, event):
        if not event.inaxes:
            return
        selected = []
        for plot, (filename, frequency) in self.plot_map.items():
            contained, info = plot.contains(event)
            if contained:
                hits = self.data \
                    .xs(filename, level='filename') \
                    .xs(frequency, level='frequency') \
                    .iloc[info['ind']]
                for (subject, level, replicate), hit in hits.iterrows():
                    analyzed_a = hit.loc['analyzed_filename', self.rater_x]
                    analyzed_b = hit.loc['analyzed_filename', self.rater_y]
                    key = (filename, frequency * 1e3, level, replicate, analyzed_a, analyzed_b)
                    selected.append(key)
        self.selected_points = selected


def main():
    parser = argparse.ArgumentParser("abr_compare")
    add_default_arguments(parser, waves=False)
    parser.add_argument('directory')
    options = parse_args(parser, waves=False)

    cols = ['filename', 'analyzed_filename', 'subject', 'frequency', 'Level', 'Replicate', 'analyzer']
    app = QtApplication()
    _, waves = options['parser'].load_analyses(options['directory'])
    waves = waves.set_index(cols).sort_index()

    presenter_a = WaveformPresenter(parser=options['parser'], interactive=False)
    presenter_b = WaveformPresenter(parser=options['parser'], interactive=False)
    presenter_c = WaveformPresenter(parser=options['parser'])
    compare = Compare(waves=waves)
    view = CompareWindow(parser=options['parser'],
                         compare=compare,
                         presenter_a=presenter_a,
                         presenter_b=presenter_b,
                         presenter_c=presenter_c
                         )
    view.show()
    app.start()
    app.stop()
