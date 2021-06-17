'''
This module defines the import/export routines for interacting with the data
store.  If you wish to customize this, simply define the load() function.

load(run_location, invert, filter) -- When the program needs a run loaded, it
will pass the run_location provided by list().  Invert is a boolean flag
indicating whether waveform polarity should be flipped.  Filter is a dictionary
containing the following keys:
    1. ftype: any of None, butterworth, bessel, etc.
    2. fh: highpass cutoff (integer in Hz)
    3. fl: lowpass cutoff (integer in Hz)
All objects of the epl.datatype.Waveform class will accept the filter
dictionary and perform the appropriate filtering.  It is recommended you use the
filtering provided by the Waveform class as the parameters of the filter will
also be recorded.  This function must return an object of the
epl.datatype.ABRSeries class.  See this class for appropriate documentation.

The save function must return a message.  If there is an error in saving, throw
the appropriate exception.
'''

import importlib
import re
from glob import glob
import os
from pathlib import Path
import time

import pandas as pd
import numpy as np

from ..datatype import Point

P_ANALYZER = re.compile('.*kHz(?:-(\w+))?-analyzed.txt')


def get_analyzer(filename):
    result = P_ANALYZER.match(filename.name).group(1)
    if result == None:
        return 'Unknown'
    return result


def waveform_string(waveform):
    data = [f'{waveform.level:.2f}']
    data.append(f'{waveform.mean(-1, 0)}')
    data.append(f'{waveform.std(-1, 0)}')
    for _, point in sorted(waveform.points.items()):
        data.append(f'{point.latency:.8f}')
        data.append(f'{point.amplitude:.8f}')
    return '\t'.join(data)


def filter_string(waveform):
    if getattr(waveform, '_zpk', None) is None:
        return 'No filtering'
    t = 'Pass %d -- z: %r, p: %r, k: %r'
    filt = [t % (i, z, p, k) for i, (z, p, k) in enumerate(waveform._zpk)]
    return '\n' + '\n'.join(filt)


def _add_replicate(x):
    x['Replicate'] = range(len(x))
    return x


def load_analysis(fname):
    th_match = re.compile(r'(?:# )?Threshold \(dB SPL\): ([-\w.]+)')
    freq_match = re.compile(r'(?:# )?Frequency \(kHz\): ([\d.]+)')
    with open(fname) as fh:
        text = fh.readline()
        th = th_match.search(text).group(1)
        th = np.nan if th == 'None' else float(th)
        text = fh.readline()
        freq = float(freq_match.search(text).group(1))

        for line in fh:
            if line.startswith('NOTE') or line.startswith('# NOTE'):
                break
        data = pd.io.parsers.read_csv(fh, sep='\t', index_col='Level')
        keep = [c for c in data.columns if not c.startswith('Unnamed')]
        data = data[keep]
        data = data.groupby('Level').apply(_add_replicate).set_index('Replicate', append=True)
    return (freq, th, data)


def parse_peaks(peaks, threshold):
    # Convert the peaks dataframe to a format that can be used by _set_points.
    pl_pattern = re.compile('P(\d) Latency')
    nl_pattern = re.compile('N(\d) Latency')
    p_latencies = {}
    n_latencies = {}

    pa_pattern = re.compile('P(\d) Amplitude')
    na_pattern = re.compile('N(\d) Amplitude')
    p_amplitudes = {}
    n_amplitudes = {}

    for i in range(5):
        if f'P{i} Amplitude' in peaks:
            p_info = pd.DataFrame({
                'x': peaks['P{i} Amplitude']
            })

    for c in peaks:
        match = pl_pattern.match(c)
        if match:
            wave = int(match.group(1))
            p_latencies[wave] = pd.DataFrame({'x': peaks[c]})
        match = nl_pattern.match(c)
        if match:
            wave = int(match.group(1))
            n_latencies[wave] = pd.DataFrame({'x': peaks[c]})

    p_latencies = pd.concat(p_latencies.values(), keys=p_latencies.keys(),
                            names=['wave'])
    p_latencies = {g: df.reset_index('Level', drop=True) \
                   for g, df in p_latencies.groupby('Level')}
    n_latencies = pd.concat(n_latencies.values(), keys=n_latencies.keys(),
                            names=['wave'])
    n_latencies = {g: df.reset_index('Level', drop=True) \
                   for g, df in n_latencies.groupby('Level')}

    for level, df in p_latencies.items():
        if level < threshold:
            df[:] = -df[:]

    for level, df in n_latencies.items():
        if level < threshold:
            df[:] = -df[:]

    return p_latencies, n_latencies


class Parser(object):

    filename_template = '{filename}-{frequency}kHz-{user}analyzed.txt'

    def __init__(self, file_format, filter_settings, user, waves=None,
                 calibration=None, latency=None):
        '''
        Parameters
        ----------
        file_format : string
            File format that will be loaded.
        filter_settings : {None, dict}
            If None, no filtering is applied. If dict, must contain ftype,
            lowpass, highpass and order as keys.
        user : {None, string}
            Person analyzing the data.
        calibration : {None, path}
            Path to calibration file
        latency : {None, path}
            Path to latency file
        '''
        self._file_format = file_format
        self._filter_settings = filter_settings
        self._user = user
        self._module_name = f'abr.parsers.{file_format}'
        self._module = importlib.import_module(self._module_name)
        self._calibration = calibration
        self._latency = latency
        self._waves = [] if waves is None else waves[:]

    def load(self, filename, frequencies=None):
        return self._module.load(filename,
                                 self._filter_settings,
                                 frequencies,
                                 calibration_file=self._calibration,
                                 latency_file=self._latency,
                                 waves=self._waves)

    def load_analysis(self, series, filename):
        freq, th, peaks = load_analysis(filename)
        series.load_analysis(th, peaks)

    def find_analyzed_files(self, filename, frequency):
        frequency = round(frequency * 1e-3, 8)
        glob_pattern = self.filename_template.format(
            filename=filename.with_suffix(''),
            frequency=frequency,
            user='*')
        path = Path(glob_pattern)
        return list(path.parent.glob(path.name))

    def get_save_filename(self, filename, frequency):
        # Round frequency to nearest 8 places to minimize floating-point
        # errors.
        user_name = self._user + '-' if self._user else ''
        frequency = round(frequency * 1e-3, 8)
        save_filename = self.filename_template.format(
            filename=filename.with_suffix(''),
            frequency=frequency,
            user=user_name)
        return Path(save_filename)

    def save(self, model):
        # Assume that all waveforms were filtered identically
        if model.meta is not None:
            meta = '\n'.join(f'# {k}: {v}' for k, v in model.meta.items())
        else:
            meta = '#'

        # Generate list of columns
        columns = ['Level', 'Avg 1msec Baseline', 'StDev 1msec Baseline']
        point_keys = sorted(model.waveforms[0].points)
        for point_number, point_type in point_keys:
            point_type_code = 'P' if point_type == Point.PEAK else 'N'
            for measure in ('Latency', 'Amplitude'):
                columns.append(f'{point_type_code}{point_number} {measure}')

        columns = '\t'.join(columns)
        spreadsheet = '\n'.join(waveform_string(w) for w in reversed(model.waveforms))
        content = CONTENT.format(threshold=model.threshold,
                                 frequency=model.freq*1e-3,
                                 columns=columns,
                                 spreadsheet=spreadsheet,
                                 metadata=meta)

        filename = self.get_save_filename(model.filename, model.freq)
        with open(filename, 'w') as fh:
            fh.writelines(content)

    def find_all(self, study_directory, frequencies=None):
        result = self._module.find_all(study_directory, self._filter_settings,
                                       frequencies=frequencies)
        if frequencies is not None:
            if np.isscalar(frequencies):
                frequencies = [frequencies]
            result = [(p, f) for (p, f) in result if f in frequencies]
        return result

    def find_processed(self, study_directory, frequencies=None):
        return [(p, f) for p, f in self.find_all(study_directory, frequencies) \
                if self.get_save_filename(p, f).exists()]

    def find_unprocessed(self, study_directory, frequencies=None):
        # k is tuple of path, frequency
        iterator = self.find_all(study_directory, frequencies=frequencies)
        return [k for k in iterator if not self.get_save_filename(*k).exists()]

    def find_analyses(self, study_directory, frequencies=None):
        analyzed = {}
        for p, f in self.find_all(study_directory, frequencies):
            analyzed[p, f] = self.find_analyzed_files(p, f)
        return analyzed

    def load_analyses(self, study_directory, frequencies=None):
        keys = []
        thresholds = []
        waves = []
        analyses = self.find_analyses(study_directory, frequencies)

        for (filename, f), analyzed_filenames in analyses.items():
            for a in analyzed_filenames:
                f, th, w = load_analysis(a)
                parts = a.stem.split('-')
                if parts[-2].endswith('kHz'):
                    analyzer = 'Unknown'
                    subject = '-'.join(parts[:-2])
                else:
                    analyzer = parts[-2]
                    subject = '-'.join(parts[:-3])

                keys.append((filename, subject, analyzer, f))
                thresholds.append(th)
                waves.append(w)

        index = pd.MultiIndex.from_tuples(keys, names=['filename', 'subject', 'analyzer', 'frequency'])
        thresholds = pd.Series(thresholds, index=index, name='thresholds').reset_index()
        waves = pd.concat(waves, keys=keys, names=['filename', 'subject', 'analyzer', 'frequency']).reset_index()
        return thresholds, waves


CONTENT = '''
# Threshold (dB SPL): {threshold:.2f}
# Frequency (kHz): {frequency:.2f}
# Version: 0.0.1
{metadata}
# NOTE: Negative latencies indicate no peak. NaN for amplitudes indicate peak was unscorable.
{columns}
{spreadsheet}
'''.strip()


PARSER_MAP = {
    'NCRAR': 'IHS text export',
}
