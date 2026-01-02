from typing import Dict, Tuple

import numpy as np
import pandas as pd
from h5py import File
from obspy import Trace, UTCDateTime
from obspy.geodetics import gps2dist_azimuth

from phasenettfps.beamforming.utils.setting import (
    LABEL_WIDTH,
    WAVEFORM_CLUSTER_LEFT_BUFFER,
    WAVEFORM_CLUSTER_RIGHT_BUFFER,
    WAVEFORM_DELTA,
)


def read_time_info(
    info_df: pd.DataFrame,
    phase_key: str = "P",
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """read the waveform phase arrival time

    Args:
        time_pd_file (str): the pd data file containing the phase arrival information
        phase_key (str): the phase key corresponding the pd file column

    Returns:
        Tuple[Dict[str, float], Dict[str, np.ndarray]]: the phase arrival info
    """

    def prepare_df(row):
        return pd.Series(
            [
                f"{row['NETWORK']}.{row['STATION']}.{row['EVENT_ID']}",
                (row["PTIME"] - row["ORIGIN_TIME"]).total_seconds(),
                (row["PSTIME"] - row["ORIGIN_TIME"]).total_seconds(),
                row["ELAT"],
                row["ELON"],
                row["EDEP"],
            ],
            index=["key", "arrival_time_p", "arrival_time_ps", "lat", "lon", "dep"],
        )

    data = info_df.apply(prepare_df, axis=1)

    arrival_times_p = {}
    arrival_times_ps = {}
    coordinates = {}
    for i in range(len(data)):
        row = data.iloc[i]
        key = row["key"]

        arrival_times_p[key] = row["arrival_time_p"]
        arrival_times_ps[key] = row["arrival_time_ps"]

        lat = float(row["lat"])
        lon = float(row["lon"])
        dep = float(row["dep"])

        coordinates[key] = np.array([lat, lon, dep])

    if phase_key == "P":
        return arrival_times_p, coordinates
    elif phase_key == "PS":
        return arrival_times_ps, coordinates


def generate_waveforms(arrival_times: Dict[str, float]) -> Dict[str, Trace]:
    # get the min and max time for all keys in arrival_times
    min_time = min(arrival_times.values())
    max_time = max(arrival_times.values())
    start_time = min_time - WAVEFORM_CLUSTER_LEFT_BUFFER
    end_time = max_time + WAVEFORM_CLUSTER_RIGHT_BUFFER
    ref_utc_time = UTCDateTime(2024, 1, 1, 0, 0, 0)
    start_time = ref_utc_time + start_time
    end_time = ref_utc_time + end_time

    # generate a gaussian waveform as a template
    label_window = np.exp(
        -((np.arange(-LABEL_WIDTH, LABEL_WIDTH + 1)) ** 2)
        / (2 * (LABEL_WIDTH / 6) ** 2)
    )

    # generate the waveforms
    waveforms = {}
    for key, arrival_time in arrival_times.items():
        arrival_time = ref_utc_time + arrival_time
        # generate the waveform
        waveform = Trace()
        waveform.stats.delta = WAVEFORM_DELTA
        waveform.stats.starttime = start_time
        waveform.data = np.zeros(int((end_time - start_time) / WAVEFORM_DELTA))

        start_pos = int((arrival_time - start_time) / WAVEFORM_DELTA)
        end_pos = start_pos + label_window.shape[0]
        waveform.data[start_pos:end_pos] = label_window
        waveforms[key] = waveform

    return waveforms


def generate_waveform_for_plot(info_df: pd.DataFrame):
    def calculate_distance(row):
        return (
            gps2dist_azimuth(row["ELAT"], row["ELON"], row["SLAT"], row["SLON"])[0]
            / 1000
        )

    info_df["DISTANCE"] = info_df.apply(calculate_distance, axis=1)
    info_df["PSTIME-PTIME"] = (info_df["PSTIME"] - info_df["PTIME"]).dt.total_seconds()
    label_window = np.exp(
        -((np.arange(-LABEL_WIDTH, LABEL_WIDTH + 1)) ** 2)
        / (2 * (LABEL_WIDTH / 6) ** 2)
    )

    # sort the dataframe by distance
    info_df = info_df.sort_values(by="DISTANCE")
    res = []
    for i in range(len(info_df)):
        row = info_df.iloc[i]
        # generate the waveform, assunme sample rate is 40Hz
        wave = np.zeros(1600, dtype=np.float32)
        # fix P arrival at 10s
        wave[400 : 400 + label_window.shape[0]] = label_window
        # PS wave is at 10s+PSTIME-PTIME
        start_pos = int((10 + row["PSTIME-PTIME"]) * 40)
        end_pos = start_pos + label_window.shape[0]
        wave[start_pos:end_pos] = label_window
        res.append([row["EVENT_ID"], row["DISTANCE"], wave, row["PSTIME-PTIME"]])
    return res


def load_waveform_for_plot(info_df: pd.DataFrame, file: File):
    def calculate_distance(row):
        return (
            gps2dist_azimuth(row["ELAT"], row["ELON"], row["SLAT"], row["SLON"])[0]
            / 1000
        )

    info_df.loc[:, "DISTANCE"] = info_df.apply(calculate_distance, axis=1)
    info_df["PSTIME-PTIME"] = (info_df["PSTIME"] - info_df["PTIME"]).dt.total_seconds()

    # sort the dataframe by distance
    info_df = info_df.sort_values(by="DISTANCE")
    res = []
    CURRENT_COMPONENT = 1
    for i in range(len(info_df)):
        row = info_df.iloc[i]
        event_id, station = row["EVENT_ID"], row["STATION"]
        entire_wave = file[str(event_id)][station][()]
        trace = Trace(data=entire_wave[CURRENT_COMPONENT])
        trace.stats.sampling_rate = 40
        trace.detrend("demean")
        trace.taper(max_percentage=0.05)
        freqmin = max(0, row["FREQ_PS"] - 2)
        freqmax = min(11, row["FREQ_PS"] + 2)
        trace.filter(
            "bandpass", freqmin=freqmin, freqmax=freqmax, corners=2, zerophase=True
        )
        trace.normalize()
        entire_wave[CURRENT_COMPONENT] = trace.data
        # entire wave is 24000 samples for 10min, we cut as 4min-10s to 4min+50s
        start_pos = (240 - 10) * 40
        end_pos = (240 + 30) * 40
        wave = entire_wave[CURRENT_COMPONENT][start_pos:end_pos]
        res.append([event_id, row["DISTANCE"], wave, row["PSTIME-PTIME"]])
    return res
