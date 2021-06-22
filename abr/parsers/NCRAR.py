import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal, stats

from abr.datatype import ABRWaveform, ABRSeries


################################################################################
# Utility functions
################################################################################
def _fix_frequency(x):
    if x == 'Click':
        return 0
    else:
        return float(x.strip(' Hz'))


def parse_identifier(identifier):
    '''
    Example: "Identifier:,IHS5453-2019AV01"

    "2019AV01" = Oct 31, 2019

    2019 = year
    A = month, where 1-9 are the Jan-Sept and A=Oct, B=Nov, C=Dec
    V = day, where 1-9 are the first days of the month, and A-V are the 10th to 31st
    01 seems to be a constant
    '''
    month_map = {}
    day_map = {}
    for i in range(1, 10):
        month_map[str(i)] = i
        day_map[str(i)] = i
    for i, code in enumerate('ABC'):
        month_map[code] = i + 10
    for i, code in enumerate('ABCDEFGHIJKLMNOPQRSTUV'):
        day_map[code] = i + 10
    system, date_code = identifier.split('-')
    year = int(date_code[:4])
    month = month_map[date_code[4]]
    day = day_map[date_code[5]]
    return pd.Series({'system': system[3:], 'date': dt.date(year, month, day)})


def _parse_line(line):
    '''
    Parse list of comma-separated values from line

    Parameters
    ----------
    line : string
        Line containing the values that need to be parsed

    Returns
    -------
    tokens : list
        List of values found in line.  If values are numeric, they will be
        converted to floats.  Otherwise they will be returned as strings.
    '''
    tokens = line.strip().split(',')[1:]
    try:
        return [float(t) for t in tokens if t]
    except ValueError:
        return [t for t in tokens if t]


def load_metadata(filename, calibration=None):
    '''
    Load the metadata stored in the ABR file

    Parameters:
    -----------
    filename : string
        Filename to load
    calibration : {None, DataFrame}
        Calibration data. If provided, will add a new column, `actual_level`.

    Returns
    -------
    info : pandas.DataFrame
        Dataframe containing information on each waveform
    '''
    info = {}
    with open(filename, 'r') as fh:
        for i, line in enumerate(fh):
            if i == 20:
                break
            name = line.split(',', 1)[0].strip(':').lower()
            info[name] = _parse_line(line)
    info = pd.DataFrame(info)

    # Number the trials.  We will use this number later to look up which column
    # contains the ABR waveform for corresponding parameter.
    info['waveform'] = np.arange(len(info))
    info.set_index('waveform', inplace=True)

    info['level'] = info['intensity']

    # Store the scaling factor for the waveform so we can recover this when
    # loading.  By default the scaling factor is 674. For 110 dB SPL, the
    # scaling factor is 337.  The statistician uses 6.74 and 3.37, but he
    # includes a division of 100 elsewhere in his code to correct.
    info['waveform_sf'] = 6.74e2

    # The rows where level is 110 dB SPL have a different scaling factor.
    info.loc[info.level >= 105, 'waveform_sf'] = 3.37e2

    # Start time of stimulus in usec (since sampling period is reported in usec,
    # we should try to be consistent with all time units).
    info['stimulus_start'] = 12.8e3

    # Interpret identifier string
    info = info.join(info['identifier'].transform(parse_identifier))

    # Load calibration data
    try:
        if calibration is not None:
            info['actual_level'] = \
                info.apply(get_actual_level, calibration=calibration, axis=1)
        else:
            info['actual_level'] = info['level']
    except Exception as e:
        raise IOError(f'Cannot load file {filename}\n{e}') from e

    return info


def load_waveforms(filename, info):
    '''
    Load the waveforms stored in the ABR file

    Only the waveforms specified in info will be loaded.  For example, if you
    have filtered the info DataFrame to only contain waveforms from channel 1,
    only those waveforms will be loaded.

    Parameters:
    -----------
    filename : string
        Filename to load
    info : pandas.DataFrame
        Waveform metadata (see `load_metadata`)

    Returns
    -------
    info : pandas.DataFrame
        Dataframe containing waveforms

    '''
    # Read the waveform table into a dataframe
    df = pd.io.parsers.read_csv(filename, skiprows=20)

    # Keep only the columns containing the signal of interest.  There are six
    # columns for each trial.  We only want the column containing the raw
    # average (i.e., not converted to uV).
    df = df[[c for c in df.columns if c.startswith('Average:')]]

    # Renumber them so we can look them up by number.  The numbers should
    # correspond to the trial number we generated in `load_metadata`.
    df.columns = np.arange(len(df.columns))

    # Loop through the entries in the info DataFrame.  This dataframe contains
    # metadata needed for processing the waveform (e.g., it tells us which
    # waveforms to keep, the scaling factor to use, etc.).
    signals = []
    for w_index, w_info in info.iterrows():
        # Compute time of each point.  Currently in usec because smp. period is
        # in usec.
        t = np.arange(len(df), dtype=np.float32)*w_info['smp. period']
        # Subtract stimulus start so that t=0 is when stimulus begins.  Convert
        # to msec.
        t = (t-w_info['stimulus_start'])*1e-3
        time = pd.Index(t, name='time')

        # Divide by the scaling factor and convert from nV to uV
        s = df[w_index]/w_info['waveform_sf']*1e-3
        s.index = time
        signals.append(s)

    # Merge together the waveforms into a single DataFrame
    waveforms = pd.concat(signals, keys=info.index, names=['waveform'])
    waveforms = waveforms.unstack(level='waveform')
    return waveforms


def is_ihs_file(filename):
    with open(filename) as fh:
        line = fh.readline()
        return line.startswith('Identifier:')


def load_calibration(calibration_file):
    calibration = pd.read_excel(calibration_file).rename(columns={
        'IHS system number': 'system',
        'IHS system booth': 'booth',
        'Calibration date': 'date',
        'Calibration frequency': 'frequency',
        'Actual level': 'measured_level',
        'Level on the IHS': 'nominal_level',
    })
    calibration['system'] = calibration['system'].astype(str)
    calibration['frequency'] = calibration['frequency'].map(_fix_frequency)
    return calibration


def get_actual_level(row, calibration):
    s = row['system']
    d = row['date']
    f = row['stim. freq.']
    l = row['level']

    matches = calibration.query('(system == @s) and (date <= @d)')
    most_recent = matches['date'].max()
    time_since_calibration = d - most_recent.date()

    if time_since_calibration.days > (6 * 30):
        raise IOError(f'No calibration within 6 months of {d.strftime("%m/%d/%Y")} on IHS system {s}.')

    result = matches.query('(date == @most_recent) and (frequency == @f) and (nominal_level == @l)')
    if result.empty:
        raise IOError(f'IHS system {s} not calibrated on {most_recent.strftime("%m/%d/%Y")} for {f} Hz {l} dB SPL (as reported by IHS).')

    if len(result) > 1:
        raise IOError(f'Duplicate calibration data on IHS system {s} for {f} Hz {l} dB SPL (as reported by IHS).')

    return result.iloc[0]['measured_level']


def get_calibration_date(system, experiment_date, calibration):
    matches = calibration.query('(system == @system) and (date <= @experiment_date)')
    most_recent_calibration = matches['date'].max().date()
    time_since_calibration = experiment_date - most_recent_calibration
    return most_recent_calibration, time_since_calibration


DEFAULT_LATENCIES = {
    1: stats.norm(3, 1),
    2: stats.norm(4, 1),
    3: stats.norm(5, 1),
    4: stats.norm(6, 1),
    5: stats.norm(7, 1),
}


def get_latencies(stim_freq, waves, latency_file):
    if latency_file is not None:
        all_latencies = pd.read_excel(latency_file, sheet_name='latencies', header=[0, 1], index_col=0)
        all_latencies = all_latencies.rename(index={'click': 0, 'Click': 0})
        all_latencies.index *= 1e3
        latencies = all_latencies.loc[stim_freq].unstack()
        latency_dict = latencies.apply(lambda x: stats.norm(x['mean'], x['std']), axis=1).to_dict()
        return {w: latency_dict[w] for w in waves}
    else:
        return {w: DEFAULT_LATENCIES[w] for w in waves}


################################################################################
# API
################################################################################
def load(filename, filter, frequencies, calibration_file, latency_file, waves,
         abr_window=8.5e-3):

    if frequencies is not None and not np.iterable(frequencies):
        frequencies = [frequencies]

    if not is_ihs_file(filename):
        raise IOError('Unsupported file format')

    if calibration_file is not None:
        calibration = load_calibration(calibration_file)
        cal_date, time_since_cal = \
            get_calibration_date( ihs_system, experiment_date, calibration)
        cal_date = cal_date.strftime('%Y%m%d'),
        time_since_cal = int(time_since_cal.days)
    else:
        calibration = None
        cal_date = 'NaT'
        time_since_cal = np.inf

    info = load_metadata(filename, calibration)

    info = info.query('channel == 1')
    fs = 1/(info.iloc[0]['smp. period']*1e-6)
    data = load_waveforms(filename, info)

    ihs_system = info.iloc[0]['system']
    experiment_date = info.iloc[0]['date']

    meta = {
        'channel': 1,
        'fs': fs,
        'filter': str(filter),
        'ihs_system': ihs_system,
        'experiment_date': experiment_date.strftime('%Y%m%d'),
        'calibration_date': cal_date,
        'days_since_calibration': time_since_cal,
    }

    series = []
    for frequency, f_info in info.groupby('stim. freq.'):
        if frequencies is not None and frequency not in frequencies:
            continue
        data = load_waveforms(filename, f_info)

        if filter is not None:
            Wn = filter['highpass']/(0.5*fs), filter['lowpass']/(0.5*fs)
            N = filter['order']
            b, a = signal.iirfilter(N, Wn)
            data[:] = signal.filtfilt(b, a, data.values, axis=0)

        data = data.query('time >= -1.5')

        waveforms = []
        for level, l_df in f_info.groupby('actual_level'):
            for replicate, (i, row) in enumerate(l_df.iterrows()):
                d = data[i]
                waveform = ABRWaveform(fs, d, level=level, replicate=replicate, channel=1)
                waveforms.append(waveform)

        latencies = get_latencies(frequency, waves, latency_file)
        s = ABRSeries(waveforms, frequency, suggested_latencies=latencies,
                      meta=meta)
        s.filename = filename
        series.append(s)
    return series


def find_all(dirname, filter_settings, frequencies=None):
    candidates = [p for p in Path(dirname).glob('**/*.txt') if 'analyzed' not in p.name]
    results = []
    for candidate in candidates:
        if is_ihs_file(candidate):
            for frequency in load_metadata(candidate)['stim. freq.'].unique():
                results.append((candidate, frequency))
    return results
