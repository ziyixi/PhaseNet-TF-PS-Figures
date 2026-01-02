"""
manuscript_manual_picks_ptf_example.py:

This script is used to generate PhaseNet-TF manual picks examples to show the performance.
"""

from pathlib import Path
from string import ascii_lowercase

import h5py
import httpx
import numpy as np
import obspy
import pygmt
import torch
import xarray as xr

from phasenettfps import resource, save_path
from phasenettfps.utils.spectrogram import GenSgram

XOFFSET = 8.6
YOFFSET = 9.6
X_SHIFTS = [1.0, XOFFSET, -XOFFSET, XOFFSET]
Y_SHIFTS = [YOFFSET + 1.8, 0, -YOFFSET, 0]
FRAMES = {
    0: {
        "waveform": ["Wsen", "xaf", "ya0.8f0.4+lAmplitude"],
        "spectrogram": ["Wsen", "xaf", "ya4f2+lFrequency"],
        "prediction": ["Wsen", "xaf", "ya0.4f0.2+lProbability"],
    },
    1: {
        "waveform": ["wsen", "xaf", "ya0.8f0.4"],
        "spectrogram": ["wsen", "xaf", "ya4f2"],
        "prediction": ["wsen", "xaf", "ya0.4f0.2"],
    },
    2: {
        "waveform": ["Wsen", "xaf", "ya0.8f0.4+lAmplitude"],
        "spectrogram": ["Wsen", "xaf", "ya4f2+lFrequency"],
        "prediction": ["WSen", "xaf+lTime (s)", "ya0.4f0.2+lProbability"],
    },
    3: {
        "waveform": ["wsen", "xaf", "ya0.8f0.4"],
        "spectrogram": ["wsen", "xaf", "ya4f2"],
        "prediction": ["wSen", "xaf+lTime (s)", "ya0.4f0.2"],
    },
}
# WAVEFORMS = [
#     ["2_52171", "C11W"],
#     ["4_54517", "A01"],
#     # ["1_51874", "A03"],
#     ["4_54566", "FONI"],
#     ["22_52416", "NMKA"],
#     # ["4_56010", "C08W"],
#     # ["2_52327", "B05W"],
#     # ["4_54675", "A14W"],
# ]
WAVEFORMS = [
    # ["11_52113", "A12W"],
    # ["4_56213", "TNGA"],
    # ["4_54845", "TNGA"],
    # ["4_55882", "TNGA"],
    ["4_54845", "TNGA", 2.62148380279541],
    ["4_54746", "NMKA", 5.341537952423096],
    ["1_51849", "A01", 2.6276421546936035],
    ["11_52113", "A12W", 3.0529417991638184],
]

# TEXT_INFO = [
#     "2010-02-17 06:32:08.00 21.56S,178.18W 452.8km YL.C11W",
#     "2010-09-26 17:20:38.87 20.78S,178.39W 618.2km YL.A01",
#     "2010-07-11 10:29:21.54 25.11S,179.80W 655.2km Z1.FONI",
#     "2010-03-07 23:30:23.92 21.94S,175.70W 148.1km Z1.NMKA",
# ]
# TEXT_INFO = [
#     "1_51874, CICA",
#     "1_51864, A14W",
#     "1_51866, F02W",
#     "1_51870, VABL",
# ]
TEXT_INFO = [
    "4_54845(2010-09-16 11:37:39.51 18.00S,177.87W 573.5km) TNGA",
    "4_54746(2010-08-16 22:09:48.45 22.47S,178.12W 366.5km) NMKA",
    "1_51849(2009-12-01 08:07:01.29 17.67S,178.56W 561.4km) A01",
    "11_52113(2009-12-02 02:25:13.61 21.00S,176.06W 189.5km) A12W",
]

MAX_CLAMP = 3


def main():
    fig = pygmt.Figure()
    pygmt.config(
        FONT_LABEL="18p",
        MAP_LABEL_OFFSET="12p",
        FONT_ANNOT_PRIMARY="11p",
        MAP_FRAME_TYPE="plain",
        MAP_TITLE_OFFSET="12p",
        FONT_TITLE="18p,black",
        MAP_FRAME_PEN="1p,black",
    )

    for iplot in range(len(X_SHIFTS)):
        fig.shift_origin(xshift=f"{X_SHIFTS[iplot]}i", yshift=f"{Y_SHIFTS[iplot]}i")
        with fig.subplot(
            nrows=7,
            ncols=1,
            subsize=(f"{5/0.618:.2f}i", "1.3i"),
            margins=["0.2i", "-0.141i"],
        ):
            (
                st,
                sgram,
                p_prediction,
                s_prediction,
                ps_prediction,
                arrivals,
                arrival_types,
                component,
            ) = load_waveform_spectrogram_prediction(
                WAVEFORMS[iplot][0], WAVEFORMS[iplot][1], WAVEFORMS[iplot][2]
            )
            for ipanel in range(3):
                with fig.set_panel(panel=ipanel):
                    fig.basemap(
                        region=[0, 120, -1, 1],
                        projection="X?i/?i",
                        frame=FRAMES[iplot]["waveform"],
                    )
                    fig.plot(
                        x=np.linspace(0, 120, 4800),
                        y=st[ipanel].data,
                        pen="1p,black",
                    )
                    plot_manual_labels(fig, arrival_types, arrivals)
                    fig.text(
                        x=3,
                        y=0.8,
                        text=f"{component[ipanel]}",
                        font="18p,Helvetica-Bold,black",
                    )
                    if ipanel == 0:
                        fig.text(
                            x=115,
                            y=0.75,
                            text=f"({ascii_lowercase[iplot]})",
                            font="30p,Helvetica-Bold,black",
                        )
                        fig.text(
                            x=55,
                            y=1.35,
                            text=TEXT_INFO[iplot],
                            font="18p,Helvetica-Bold,black",
                            no_clip=True,
                        )
            pygmt.makecpt(
                cmap="jet", series=[0, MAX_CLAMP, 0.1], continuous=True, reverse=False
            )
            for ipanel in range(3, 6):
                with fig.set_panel(panel=ipanel):
                    fig.basemap(
                        region=[0, 120, 0, 10],
                        projection="X?i/?i",
                        frame=FRAMES[iplot]["spectrogram"],
                    )
                    fig.grdimage(
                        sgram[ipanel - 3],
                        cmap=True,
                    )
                    plot_manual_labels(fig, arrival_types, arrivals)
                    fig.text(
                        x=3,
                        y=9,
                        text=f"{component[ipanel-3]}",
                        font="18p,Helvetica-Bold,white",
                    )
            with fig.set_panel(panel=6):
                fig.basemap(
                    region=[0, 120, 0, 1],
                    projection="X?i/?i",
                    frame=FRAMES[iplot]["prediction"],
                )
                fig.plot(
                    x=np.linspace(0, 120, len(p_prediction)),
                    y=p_prediction,
                    pen="1p,red",
                    label="Predicted P",
                )
                fig.plot(
                    x=np.linspace(0, 120, len(s_prediction)),
                    y=s_prediction,
                    pen="1p,magenta",
                    label="Predicted S",
                )
                fig.plot(
                    x=np.linspace(0, 120, len(p_prediction)),
                    y=ps_prediction,
                    pen="1p,green",
                    label="Predicted PS",
                )
                plot_manual_labels(fig, arrival_types, arrivals, with_label=True)
                fig.legend(
                    position="JTR+jTR+o0.2c/0.2c",
                    box="+gwhite+p1p,black",
                )

    fig.shift_origin(xshift=f"{-XOFFSET/2+1}i")
    pygmt.makecpt(
        cmap="jet", series=[0, MAX_CLAMP, 0.1], continuous=True, reverse=False
    )
    fig.colorbar(
        position="JBC+w5i/0.8c+h+o0i/1.8c",
        box=False,
        frame=["a0.5f", f'"+LSpecrogram Amplitude"'],
        scale=1,
    )

    save_path(fig, Path(__file__).resolve().stem)


def load_waveform_spectrogram_prediction(event_id, station_id, ps_center_freq):
    waveform_h5 = resource(["waveforms", "waveform.h5"], normal_path=True)
    raw = h5py.File(waveform_h5, "r")[event_id][station_id]
    raw_wave = raw[:, 9200:14000]

    # * waveform
    st = obspy.Stream()
    for i in range(len(raw_wave)):
        st.append(obspy.Trace(data=raw_wave[i], header={"delta": 0.025}))

    st_filter = st.copy()
    st_no_filter = st.copy()

    st_filter.detrend("linear")
    st_filter.taper(max_percentage=0.01)
    st_filter.filter("bandpass", freqmin=1, freqmax=10, corners=2, zerophase=True)
    st_filter.normalize(global_max=True)
    st_no_filter.normalize(global_max=True)

    arrivals = raw.attrs["phase_index"]
    arrivals = [item - 9200 for item in arrivals]
    arrival_types = raw.attrs["phase_type"]
    component = raw.attrs["component"]

    # * spectrogram
    sgram_gen = GenSgram(max_clamp=MAX_CLAMP)
    # wrap obspy wave into torch tensor with batch==1
    wave_filter = np.zeros((3, 4800), dtype=np.float32)
    wave_no_filter = np.zeros((3, 4800), dtype=np.float32)
    for i in range(3):
        wave_filter[i, :] = st_filter[i].data
        wave_no_filter[i, :] = st_no_filter[i].data
    wave_filter = torch.from_numpy(wave_filter).unsqueeze(0)
    wave_no_filter = torch.from_numpy(wave_no_filter).unsqueeze(0)
    sgram = sgram_gen(wave_filter)
    sgram = sgram.squeeze(0).numpy()
    # sgram is with shape 3, 64, 4800, i.e., channel, freq, time, wrap it to xarray
    sgram = xr.DataArray(
        sgram,
        dims=["channel", "freq", "time"],
        coords={
            "channel": ["1", "2", "Z"],
            "freq": np.linspace(0, 10, 64),
            "time": np.linspace(0, 120, 4800),
        },
    )

    # * prediction
    id = f"{event_id}.{station_id}"
    timestamp = "1970-01-01T00:00:00.000000"
    w = wave_no_filter.squeeze(0).numpy().tolist()
    sensitivity = 0.5
    return_prediction = True
    request_body = {
        "id": id,
        "timestamp": timestamp,
        "waveform": w,
        "sensitivity": sensitivity,
        "return_prediction": return_prediction,
    }
    url = "http://0.0.0.0:8081/api/predict"
    headers = {"Content-Type": "application/json"}
    response = httpx.post(url, headers=headers, json=request_body, timeout=6000)
    if response.status_code == 200:
        print("success", id)
    else:
        print("fail", id)
        raise Exception(response.text)
    p_prediction = np.array(response.json()["prediction"]["P"][:4800])
    s_prediction = np.array(response.json()["prediction"]["S"][:4800])
    ps_prediction = np.array(response.json()["prediction"]["PS"][:4800])

    return (
        st_filter,
        sgram,
        p_prediction,
        s_prediction,
        ps_prediction,
        arrivals,
        arrival_types,
        component,
    )


def plot_manual_labels(fig, arrival_types, arrivals, with_label=False):
    colors = ["red", "magenta", "green"]
    for iphase, phase in enumerate(["P", "S", "PS"]):
        for itype, type in enumerate(arrival_types):
            if type == phase:
                fig.plot(
                    x=[
                        arrivals[itype] * 0.025,
                        arrivals[itype] * 0.025,
                    ],
                    y=[-20, 20],
                    pen=f"2p,{colors[iphase]},.-.",
                    label=f"Manually picked {phase}" if with_label else None,
                )
