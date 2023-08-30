"""
Collection of classes for handling common data types
"""
from enum import Enum
import functools
import operator

import numpy as np
import pandas as pd
from scipy import signal

from atom.api import Atom, Bool, Int, Property, Typed, Value

from .peakdetect import (generate_latencies_bound, generate_latencies_skewnorm,
                         guess, guess_iter, peak_iterator)


@functools.total_ordering
class Point(Enum):

    PEAK = 'peak'
    VALLEY = 'valley'

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplementedError


class ABRWaveform:

    def __init__(self, fs, signal, level, replicate=0, channel=1):
        self.fs = fs
        self.signal = signal
        self.level = level
        self.replicate = replicate
        self.channel = channel
        self.points = {}
        self.series = None

    @property
    def x(self):
        return self.signal.index.values

    @property
    def y(self):
        return signal.detrend(self.signal.values)

    def is_subthreshold(self):
        if self.series.threshold is None or np.isnan(self.series.threshold):
            return False
        return self.level < self.series.threshold

    def is_suprathreshold(self):
        if self.series.threshold is None or np.isnan(self.series.threshold):
            return True
        return self.level >= self.series.threshold

    def stat(self, lb, ub, func):
        return func(self.signal.loc[lb:ub])

    def mean(self, lb, ub):
        return self.stat(lb, ub, np.mean)

    def std(self, lb, ub):
        return self.stat(lb, ub, np.std)

    def set_point(self, wave, ptype, index=None, latency=None,
                  unscorable=False):

        # First, figure out index given requested latency
        if index is None and latency is None:
            raise ValueError('Must provide index or latency')
        elif index is not None and latency is not None:
            raise ValueError('Must provide either index or latency')
        elif latency is not None:
            index = np.searchsorted(self.x, latency)

        # Now, create point if it does not exist
        if (wave, ptype) not in self.points:
            point = WaveformPoint(self, 0, wave, ptype)
            self.points[wave, ptype] = point

        # Update the values on the point
        self.points[wave, ptype].index = int(index)
        self.points[wave, ptype].unscorable = unscorable

    def clear_points(self):
        self.points = {}

    def clear_peaks(self):
        for wave, ptype in list(self.points):
            if ptype == Point.PEAK:
                del self.points[wave, ptype]

    def clear_valleys(self):
        for wave, ptype in list(self.points):
            if ptype == Point.VALLEY:
                del self.points[wave, ptype]

    def set_points(self, guesses, ptype):
        for wave, wave_guess in guesses.iterrows():
            index = wave_guess.get('index', np.nan)
            if np.isfinite(index):
                self.set_point(wave, ptype, index=int(index))
            else:
                self.set_point(wave, ptype, latency=wave_guess['x'])


class WaveformPoint(Atom):
    '''
    Parameters
    ----------
    TODO
    '''
    parent = Typed(ABRWaveform)
    _index = Int()
    index = Property()
    wave_number = Int()
    point_type = Typed(Point)
    iterator = Value()
    unscorable = Bool(False)

    def __init__(self, parent, index, wave_number, point_type):
        # Order of setting attributes is important here
        self.parent = parent
        self.point_type = point_type
        self.wave_number = wave_number
        invert = self.is_valley()
        iterator = peak_iterator(parent, index, invert=invert)
        next(iterator)
        self.iterator = iterator
        self.index = index

    def _observe__index(self, event):
        if event['type'] == 'update':
            self.iterator.send(('set', event['value']))

    @property
    def x(self):
        return self.parent.x[self.index]

    @property
    def y(self):
        return self.parent.y[self.index]

    def is_peak(self):
        return self.point_type == Point.PEAK

    def is_valley(self):
        return self.point_type == Point.VALLEY

    def _get_index(self):
        return self._index

    def _set_index(self, index):
        # This makes sure we cannot have negative latencies
        if self.parent.x[index] < 0:
            index = np.searchsorted(self.parent.x, 0)
        self._index = int(np.clip(index, 0, len(self.parent.x)-1))

    @property
    def latency(self):
        latency = self.x
        if self.parent.is_subthreshold():
            return -np.abs(latency)
        elif self.unscorable:
            return -np.abs(latency)
        return latency

    @property
    def amplitude(self):
        if self.unscorable:
            return np.nan
        return self.parent.signal.iloc[self.index]

    def move(self, step):
        self.index = self.iterator.send(step)

    def time_to_index(self, time):
        return np.searchsorted(self.parent.x, time)


class ABRSeries(object):

    def __init__(self, waveforms, freq, suggested_latencies, threshold=np.nan,
                 meta=None):

        waveforms.sort(key=operator.attrgetter('level'))
        self.waveforms = waveforms
        self.freq = freq
        self.threshold = threshold
        self.meta = meta
        self.suggested_latencies = suggested_latencies
        for waveform in self.waveforms:
            waveform.series = self

    def get_level(self, level):
        for waveform in self.waveforms:
            if waveform.level == level:
                return waveform
        raise AttributeError(f'{level} dB SPL not in series')

    def guess_p(self, latencies):
        guesses = guess_iter(self.waveforms, latencies)
        self.set_points(guesses, Point.PEAK)

    def guess_n(self):
        n_latencies = {}
        for w in self.waveforms:
            g = {p.wave_number: p.x for p in w.points.values() if p.is_peak()}
            g = pd.DataFrame({'x': g})
            n_priors = generate_latencies_bound(g)
            if 5 in n_priors:
                n_priors_skew = generate_latencies_skewnorm(g)
                n_priors[5] = n_priors_skew[5]
            n_latencies[w] = n_priors
        level_guesses = guess(self.waveforms, n_latencies, invert=True)
        self.set_points(level_guesses, Point.VALLEY)

    def update_guess(self, waveform, point):
        p = waveform.points[point]
        g = {p.wave_number: p.x}
        g = pd.DataFrame({'x': g})
        latencies = generate_latencies_skewnorm(g)
        i = self.waveforms.index(waveform)
        waveforms = self.waveforms[:i]
        level_guesses = guess_iter(waveforms, latencies, invert=p.is_valley())
        self.set_points(level_guesses, p.point_type)

    def clear_points(self):
        for waveform in self.waveforms:
            waveform.clear_points()

    def clear_peaks(self):
        for waveform in self.waveforms:
            waveform.clear_peaks()

    def clear_valleys(self):
        for waveform in self.waveforms:
            waveform.clear_valleys()

    def set_points(self, guesses, ptype):
        for waveform, guess in guesses.items():
            waveform.set_points(guess, ptype)

    def load_analysis(self, threshold, points):
        if threshold is None:
            threshold = np.nan
        self.threshold = threshold
        for j, waveform in enumerate(self.waveforms[::-1]):
            analysis = points.iloc[j]
            for i in range(1, 6):
                try:
                    p_latency = np.abs(analysis[f'P{i} Latency'])
                    n_latency = np.abs(analysis[f'N{i} Latency'])
                    p_amplitude = analysis[f'P{i} Amplitude']
                    n_amplitude = analysis[f'N{i} Amplitude']
                    p_unscorable = bool(np.isnan(p_amplitude))
                    n_unscorable = bool(np.isnan(n_amplitude))
                    waveform.set_point(i, Point.PEAK, latency=p_latency, unscorable=p_unscorable)
                    waveform.set_point(i, Point.VALLEY, latency=n_latency, unscorable=n_unscorable)
                except KeyError:
                    pass
