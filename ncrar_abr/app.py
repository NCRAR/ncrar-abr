import argparse
from collections import Counter
import json
from pathlib import Path

from matplotlib import pylab as pl
from numpy import random
import pandas as pd
from scipy import stats

import enaml
from enaml.application import deferred_call
from enaml.qt.qt_application import QtApplication
from enaml.qt.QtCore import QStandardPaths

with enaml.imports():
    from enaml.stdlib.message_box import information
    from ncrar_abr.compare import Compare
    from ncrar_abr.compare_window import CompareWindow
    from ncrar_abr.launch_window import LaunchWindow, Settings
    from ncrar_abr.main_window import (DNDWindow, load_files, SerialWindow)
    from ncrar_abr.presenter import SerialWaveformPresenter, WaveformPresenter


from ncrar_abr import parsers
from ncrar_abr.parsers import Parser


def config_path():
    config_path = Path(QStandardPaths.standardLocations(QStandardPaths.GenericConfigLocation)[0])
    return config_path / 'NCRAR' / 'abr'

def config_file():
    config_file =  config_path() / 'config.json'
    config_file.parent.mkdir(exist_ok=True, parents=True)
    return config_file


def read_config():
    filename = config_file()
    if not filename.exists():
        return {}
    return json.loads(filename.read_text())


def write_config(config):
    filename = config_file()
    filename.write_text(json.dumps(config, indent=2))


def add_default_arguments(parser, waves=True):
    parser.add_argument('--nofilter', action='store_false', dest='filter',
                        default=True, help='Do not filter waveform')
    parser.add_argument('--lowpass',
                        help='Lowpass cutoff (Hz), default 3000 Hz',
                        default=3000, type=float)
    parser.add_argument('--highpass',
                        help='Highpass cutoff (Hz), default 300 Hz',
                        default=300, type=float)
    parser.add_argument('--order',
                        help='Filter order, default 1st order', default=1,
                        type=int)
    parser.add_argument('--parser', default='HDF5', help='Parser to use')
    parser.add_argument('--user', help='Name of person analyzing data')
    parser.add_argument('--calibration', help='Calibration file')
    parser.add_argument('--latency', help='Latency file')
    if waves:
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--threshold-only', action='store_true')
        group.add_argument('--all-waves', action='store_true')
        group.add_argument('--waves', type=int, nargs='+')


def parse_args(parser, waves=True):
    options = parser.parse_args()
    exclude = ('filter', 'lowpass', 'highpass', 'order', 'parser', 'user',
               'waves', 'all_waves', 'threshold_only')
    new_options = {k: v for k, v in vars(options).items() if k not in exclude}
    filter_settings = None
    if options.filter:
        filter_settings = {
            'lowpass': options.lowpass,
            'highpass': options.highpass,
            'order': options.order,
        }

    if waves:
        if options.all_waves:
            waves = [1, 2, 3, 4, 5]
        elif options.threshold_only:
            waves = []
        else:
            waves = options.waves[:]
    else:
        waves = []

    new_options['parser'] = Parser(file_format=options.parser,
                                   filter_settings=filter_settings,
                                   user=options.user,
                                   calibration=options.calibration,
                                   waves=waves,
                                   latency=options.latency)
    return new_options


def main_launcher():
    parser = argparse.ArgumentParser('ncrar-abr')
    args = parser.parse_args()

    app = QtApplication()
    settings = Settings()
    settings.set_state(read_config())
    window = LaunchWindow(settings=settings)
    window.show()
    app.start()
    app.stop()
    write_config(settings.get_state())


def main_gui():
    parser = argparse.ArgumentParser('ncrar-abr-gui')
    add_default_arguments(parser)
    parser.add_argument('--demo', action='store_true', dest='demo',
                        default=False, help='Load demo data')
    parser.add_argument('filenames', nargs='*')
    options = parse_args(parser)

    app = QtApplication()
    view = DNDWindow(parser=options['parser'])

    filenames = [(Path(f), None) for f in options['filenames']]

    deferred_call(load_files, options['parser'], filenames, view.find('dock_area'))

    view.show()
    app.start()
    app.stop()


def main_batch():
    parser = argparse.ArgumentParser("ncrar-abr-batch")
    add_default_arguments(parser)
    parser.add_argument('dirnames', nargs='*')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--skip-errors', action='store_true')
    parser.add_argument('--frequencies', nargs='*', type=float)
    parser.add_argument('--shuffle', action='store_true')
    options = parse_args(parser)
    parser = options['parser']

    unprocessed = []
    for dirname in options['dirnames']:
        files = parser.find_unprocessed(dirname, frequencies=options['frequencies'])
        unprocessed.extend(files)

    if options['shuffle']:
        random.shuffle(unprocessed)

    if options['list']:
        counts = Counter(f for f, _ in unprocessed)
        for filename, n in counts.items():
            filename = filename.stem
            print(f'{filename} ({n})')
        return

    app = QtApplication()
    if len(unprocessed) == 0:
        information(None, 'Data', 'No datasets to process.')
        return

    presenter = SerialWaveformPresenter(parser=parser, unprocessed=unprocessed)
    view = SerialWindow(presenter=presenter)
    view.show()
    app.start()
    app.stop()


def aggregate(study_directory, output_file):
    output_file = Path(output_file).with_suffix('.xlsx')
    study_directory = Path(study_directory)
    analyzed = list(study_directory.glob('*analyzed*.txt'))

    keys = []
    thresholds = []
    waves = []
    for a in analyzed:
        f, _, w = parsers.load_analysis(a)

        parts = a.stem.split('-')
        if parts[-2].endswith('kHz'):
            analyzer = 'Unknown'
            subject = '-'.join(parts[:-2])
        else:
            analyzer = parts[-2]
            subject = '-'.join(parts[:-3])

        keys.append((subject, analyzer, f))
        waves.append(w)

    index = pd.MultiIndex.from_tuples(keys, names=['subject', 'analyzer', 'frequency'])
    waves = pd.concat(waves, keys=keys, names=['subject', 'analyzer', 'frequency']).reset_index()
    for i in range(1, 7):
        try:
            waves[f'W{i} Amplitude'] = waves[f'P{i} Amplitude'] - waves[f'N{i} Amplitude']
            waves[f'W{i} Amplitude re baseline'] = waves[f'P{i} Amplitude'] - waves[f'1msec Avg']
        except KeyError:
            pass

    cols = ['frequency'] + [c for c in waves.columns if c.startswith('P') and c.endswith('Latency')]
    latencies = waves[cols].copy()
    latency_summary = latencies.rename(columns={
        'frequency': 'stimulus',
        'P1 Latency': 1,
        'P2 Latency': 2,
        'P3 Latency': 3,
        'P4 Latency': 4,
        'P5 Latency': 5,
    }).groupby('stimulus').agg(['mean', 'std'])

    with pd.ExcelWriter(output_file) as writer:
        waves.to_excel(writer, sheet_name='waves', index=False)
        latency_summary.to_excel(writer, sheet_name='latencies')


def main_aggregate():
    parser = argparse.ArgumentParser('ncrar-abr-aggregate')
    parser.add_argument('study_directory')
    parser.add_argument('output_file')
    args = parser.parse_args()
    aggregate(args.study_directory, args.output_file)


def make_shortcuts():
    from importlib.resources import files
    import os
    import sys
    from pyshortcuts import make_shortcut, platform
    bindir = 'bin'
    if platform.startswith('win'):
        bindir = 'Scripts'

    icon_file = files('ncrar_abr').joinpath('abr-icon.ico')

    shortcut = make_shortcut(
        os.path.normpath(os.path.join(sys.prefix, bindir, 'ncrar-abr')),
        name='Auditory Wave Analysis',
        folder='NCRAR',
        description='Auditroy Wave Analysis customized for NCRAR',
        icon=icon_file,
        terminal=False,
        desktop=False,
        startmenu=True,
    )


def main_compare():
    parser = argparse.ArgumentParser("ncrar-abr-compare")
    add_default_arguments(parser, waves=False)
    parser.add_argument('directory')
    options = parse_args(parser, waves=False)

    cols = ['filename', 'analyzed_filename', 'subject', 'frequency', 'Level', 'Replicate', 'Channel', 'analyzer']
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
                         presenter_c=presenter_c,
                         )
    view.show()
    app.start()
    app.stop()
