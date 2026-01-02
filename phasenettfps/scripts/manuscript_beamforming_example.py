"""
manuscript_beamforming_example.py:

This script is the figure to illustrate the beamforming method in the manuscript.
"""

import pickle
from collections import defaultdict
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pygmt
import xarray as xr
from h5py import File

from phasenettfps import resource, save_path
from phasenettfps.beamforming.core.bf import FreqBF
from phasenettfps.beamforming.utils.io import (
    generate_waveforms,
    load_waveform_for_plot,
    read_time_info,
)
from phasenettfps.beamforming.utils.optim import brute_force_search
from phasenettfps.beamforming.utils.theoritical import get_theoritical_azi_v_takeoff

REGION = [-180, -174, -21, -16]
SCALE = 1.2
WAVE_SCALE = 1.5

CENTER_LON = 360 - 177.96
CENTER_LAT = -17.94
CENTER_DEP = 561.31
STATION_LLD = (-174.63, -20.28)
STATION_NAME = "FONI"
# CENTER_LON = 360 - 175.69
# CENTER_LAT = -20.09
# CENTER_DEP = 216.21
# STATION_LLD = (-175.62, -21.53)


def load_info():
    # * load the dfs
    arrival_info = pd.read_csv(
        resource(
            ["mcmc", "mcmcbf_manual_filtered__v_takeoff_1_3_15_45", "arrival_info.csv"],
            normal_path=True,
        ),
        index_col=0,
    )
    arrival_info["ORIGIN_TIME"] = pd.to_datetime(arrival_info["ORIGIN_TIME"])
    arrival_info["PTIME"] = pd.to_datetime(arrival_info["PTIME"])
    arrival_info["PSTIME"] = pd.to_datetime(arrival_info["PSTIME"])
    # * load ps filter range info
    raw_catalog = pd.read_csv(
        resource(
            ["catalog", "tonga_picks_updated_2023_0426.csv"],
            normal_path=True,
        ),
    )

    filter_info = {}

    def update_kernel(row):
        filter_info[(row["originid"], row["stacode"])] = row["frequency_PS"]

    raw_catalog.apply(update_kernel, axis=1)
    arrival_info["FREQ_PS"] = arrival_info.apply(
        lambda row: filter_info[(row["EVENT_ID"], row["STATION"])], axis=1
    )

    MCMC_DIR = Path(
        resource(
            ["mcmc", "mcmcbf_manual_filtered__v_takeoff_1_3_15_45"], normal_path=True
        )
    )
    mcmc_res = defaultdict(list)
    for file in MCMC_DIR.glob("*.nc"):
        trace = az.from_netcdf(file)
        df = az.summary(trace, round_to=2)
        with open(MCMC_DIR / f"idx2trueidx_{file.stem.split('_')[-1]}.pkl", "rb") as f:
            mapper = pickle.load(f)

        for i, row in df.iterrows():
            mcmc_res[mapper[int(i[2:-1])]].append(row["mean"])
    mcmc_keep = set()
    for k, v in mcmc_res.items():
        if np.mean(v) >= 0.95 and np.max(v) - np.min(v) <= 0.1:
            mcmc_keep.add(k)
    arrival_info_after_mcmc = arrival_info[arrival_info["INDEX"].isin(mcmc_keep)]

    # * now we do filter to the box, horizontally it's 1 degrees, vertically it's 100 km
    before_mcmc = arrival_info[
        (arrival_info["ELAT"] > CENTER_LAT - 0.5 * SCALE)
        & (arrival_info["ELAT"] < CENTER_LAT + 0.5 * SCALE)
        & (arrival_info["ELON"] > CENTER_LON - 0.5 * SCALE)
        & (arrival_info["ELON"] < CENTER_LON + 0.5 * SCALE)
        & (arrival_info["EDEP"] > CENTER_DEP - 50 * SCALE)
        & (arrival_info["EDEP"] < CENTER_DEP + 50 * SCALE)
        & (arrival_info["STATION"] == STATION_NAME)
    ]
    after_mcmc = arrival_info_after_mcmc[
        (arrival_info_after_mcmc["ELAT"] > CENTER_LAT - 0.5 * SCALE)
        & (arrival_info_after_mcmc["ELAT"] < CENTER_LAT + 0.5 * SCALE)
        & (arrival_info_after_mcmc["ELON"] > CENTER_LON - 0.5 * SCALE)
        & (arrival_info_after_mcmc["ELON"] < CENTER_LON + 0.5 * SCALE)
        & (arrival_info_after_mcmc["EDEP"] > CENTER_DEP - 50 * SCALE)
        & (arrival_info_after_mcmc["EDEP"] < CENTER_DEP + 50 * SCALE)
        & (arrival_info_after_mcmc["STATION"] == STATION_NAME)
    ]
    # before_mcmc_waveforms = generate_waveform_for_plot(before_mcmc)
    # after_mcmc_waveforms = generate_waveform_for_plot(after_mcmc)
    waveform_h5 = File(resource(["waveforms", "waveform.h5"], normal_path=True), "r")
    before_mcmc_waveforms = load_waveform_for_plot(before_mcmc, waveform_h5)
    after_mcmc_waveforms = load_waveform_for_plot(after_mcmc, waveform_h5)

    before_mcmc_arrival_times_p, before_mcmc_coordinates_p = read_time_info(
        before_mcmc, phase_key="P"
    )
    before_mcmc_arrival_times_ps, before_mcmc_coordinates_ps = read_time_info(
        before_mcmc, phase_key="PS"
    )
    after_mcmc_arrival_times_ps, after_mcmc_coordinates_ps = read_time_info(
        after_mcmc, phase_key="PS"
    )

    # * now we perform the beamforming
    rrange = {
        "phi": np.arange(-90, 90, 2),
        "theta": np.arange(0, 360, 2),
        "v": np.arange(5.5, 11.5, 0.1),
    }

    def perform_beam_forming(arrival_times, coordinates, desc_info):
        waveforms = generate_waveforms(arrival_times)
        m = len(waveforms)
        n = len(waveforms[list(waveforms.keys())[0]].data)
        waves = np.zeros((m, n), dtype=np.float64)
        coors = np.zeros((m, 3), dtype=np.float64)
        for idx, k in enumerate(waveforms):
            waves[idx, :] = waveforms[k].data[:n]
            coors[idx, :] = coordinates[k]
        bf = FreqBF(waves, coors)

        (
            index_takeoff_opt,
            index_azi_opt,
            index_v_opt,
            amplitude_opt,
            opt_array,
        ) = brute_force_search(
            bf,
            rrange["phi"],
            rrange["theta"],
            rrange["v"],
            f"Optim search... for {desc_info}",
        )

        (
            azi_theoritical,
            v_theoritical,
            takeoff_theoritical,
        ) = get_theoritical_azi_v_takeoff(coors, STATION_LLD)

        res = {
            "phi_opt": rrange["phi"][index_takeoff_opt],
            "theta_opt": rrange["theta"][index_azi_opt],
            "v_opt": rrange["v"][index_v_opt],
            "azi_theoritical": azi_theoritical,
            "v_theoritical": v_theoritical,
            "takeoff_theoritical": takeoff_theoritical,
        }
        res.update(
            {
                "fix_phi_theta": opt_array[index_takeoff_opt, index_azi_opt, :],
                "fix_phi_v": opt_array[index_takeoff_opt, :, index_v_opt],
                "fix_theta_v": opt_array[:, index_azi_opt, index_v_opt],
                "fix_phi": opt_array[index_takeoff_opt, :, :],
                "fix_theta": opt_array[:, index_azi_opt, :],
                "fix_v": opt_array[:, :, index_v_opt],
            }
        )
        return res

    res_before_mcmc_p = perform_beam_forming(
        before_mcmc_arrival_times_p, before_mcmc_coordinates_p, "before mcmc P"
    )
    res_before_mcmc_ps = perform_beam_forming(
        before_mcmc_arrival_times_ps, before_mcmc_coordinates_ps, "before mcmc PS"
    )
    res_after_mcmc_ps = perform_beam_forming(
        after_mcmc_arrival_times_ps, after_mcmc_coordinates_ps, "after mcmc PS"
    )
    return (
        res_before_mcmc_p,
        res_before_mcmc_ps,
        res_after_mcmc_ps,
        before_mcmc_waveforms,
        after_mcmc_waveforms,
        before_mcmc,
        after_mcmc,
    )


# * =========================Helper functions=========================
def plot_earth_relief(fig: pygmt.Figure):
    grd_topo = pygmt.datasets.load_earth_relief(
        resolution="02m", region=REGION, registration="gridline"
    )
    assert type(grd_topo) == xr.DataArray
    # plot 2000m contour of topography, start from -2000m to -10000m
    fig.grdcontour(
        grd_topo,
        interval=1000,
        pen="0.8p,gray",
        limit="-10000/-7000",
    )
    # plot only -1000m contour
    fig.grdcontour(
        grd_topo,
        interval=1000,
        pen="0.5p,gray",
        limit="-1100/-1000",
    )
    fig.coast(land="gray")


def plot_text(fig: pygmt.Figure):
    text_elements = [
        {
            "text": "Tonga Trench",
            "x": -173.3,
            "y": -22.5,
            "font": "10p,Helvetica-Bold,black",
            "angle": 65,
        },
        {
            "text": "Lau Basin",
            "x": -176,
            "y": -18,
            "font": "10p,Helvetica-Bold,black",
            "angle": 65,
        },
        {
            "text": "Fiji",
            "x": 178,
            "y": -17,
            "font": "10p,Helvetica-Bold,black",
            "angle": 0,
        },
    ]

    for element in text_elements:
        fig.text(**element)


def plot_inset(fig: pygmt.Figure):
    with fig.inset(position="jBL+w7c+o0.2c", margin=0):
        fig.coast(
            region="g",
            projection=f"W{(REGION[0]+REGION[1])//2}/3c",
            land="gray",
            water="white",
            frame=["wsen", "xafg", "yafg"],
        )
        fig.plot(
            x=[REGION[0], REGION[1], REGION[1], REGION[0], REGION[0]],
            y=[REGION[2], REGION[2], REGION[3], REGION[3], REGION[2]],
            pen="0.5p,black",
            projection=f"W{(REGION[0]+REGION[1])//2}/3c",
        )


def plot_arrow(fig):
    style = "V1c+e+h0"
    fig.plot(
        x=[-173.5],
        y=[-25],
        style=style,
        direction=([-65], [2]),
        pen="2p,black",
        fill="black",
    )


# * =========================Main function=========================


def main():
    fig = pygmt.Figure()
    pygmt.config(
        FONT_LABEL="auto",
        MAP_LABEL_OFFSET="12p",
        FONT_ANNOT_PRIMARY="auto",
        MAP_FRAME_TYPE="plain",
        MAP_TITLE_OFFSET="12p",
        FONT_TITLE="18p,black",
        MAP_FRAME_PEN="1p,black",
    )

    (
        res_before_mcmc_p,
        res_before_mcmc_ps,
        res_after_mcmc_ps,
        before_mcmc_waveforms,
        after_mcmc_waveforms,
        before_mcmc_df,
        after_mcmc_df,
    ) = load_info()
    after_waveforms_event_id = set([item[0] for item in after_mcmc_waveforms])
    waveform_plot_array = []
    for iwave, (event_id, distance, wave, ps_time_diff) in enumerate(
        before_mcmc_waveforms
    ):
        if event_id in after_waveforms_event_id:
            waveform_plot_array.append(
                [event_id, "red", wave * WAVE_SCALE + distance, ps_time_diff, distance]
            )
        else:
            waveform_plot_array.append(
                [
                    event_id,
                    "black",
                    wave * WAVE_SCALE + distance,
                    ps_time_diff,
                    distance,
                ]
            )

    # * prepare images
    # prepare 2d images
    images_2d = []
    all_res = [res_before_mcmc_p, res_before_mcmc_ps, res_after_mcmc_ps]
    for each_res in all_res:
        images_2d.append(
            [
                xr.DataArray(
                    each_res["fix_v"],
                    dims=("phi", "theta"),
                    coords={
                        "theta": np.arange(0, 360, 2),
                        "phi": np.arange(-90, 90, 2),
                    },
                ),
                xr.DataArray(
                    each_res["fix_theta"],
                    dims=("phi", "v"),
                    coords={
                        "phi": np.arange(-90, 90, 2),
                        "v": np.arange(5.5, 11.5, 0.1),
                    },
                ),
                xr.DataArray(
                    each_res["fix_phi"],
                    dims=("theta", "v"),
                    coords={
                        "theta": np.arange(0, 360, 2),
                        "v": np.arange(5.5, 11.5, 0.1),
                    },
                ),
            ]
        )
    # normalize images_2d to 0-1
    for i in range(3):
        for j in range(3):
            images_2d[i][j] = (images_2d[i][j] - images_2d[i][j].min()) / (
                images_2d[i][j].max() - images_2d[i][j].min()
            )

    # generate images1d
    images_1d_y = []
    images_1d_x = []
    images_1d_expected = []
    for each_res in all_res:
        images_1d_y.append(
            [
                each_res["fix_phi_theta"],
                each_res["fix_phi_v"],
                each_res["fix_theta_v"],
            ]
        )
        images_1d_x.append(
            [
                np.arange(5.5, 11.5, 0.1),
                np.arange(0, 360, 2),
                np.arange(-90, 90, 2),
            ]
        )
        images_1d_expected.append(
            [
                each_res["v_theoritical"],
                each_res["azi_theoritical"],
                each_res["takeoff_theoritical"],
            ]
        )

    # shift to top and right a little bit to avoid cut-off
    fig.shift_origin(xshift="1i", yshift="0.5i")

    # * 1. Plot the map
    fig.shift_origin(yshift="10i")
    fig.basemap(
        region=REGION,
        projection="M3i",
        frame=["WSen", "xaf", "yaf"],
    )
    fig.text(
        x=-179.5,
        y=-16.5,
        text="(a)",
        font="14p,Helvetica-Bold,black",
        justify="LM",
        no_clip=True,
    )
    plot_earth_relief(fig)
    # plot_text(fig)
    # plot_inset(fig)
    # plot_arrow(fig)
    fig.plot(
        x=before_mcmc_df["ELON"],
        y=before_mcmc_df["ELAT"],
        style="a0.2c",
        fill="black",
        label="Removed",
    )
    fig.plot(
        x=after_mcmc_df["ELON"],
        y=after_mcmc_df["ELAT"],
        style="a0.2c",
        fill="red",
        label="Kept",
    )
    fig.plot(
        x=STATION_LLD[0],
        y=STATION_LLD[1],
        style="t0.3c",
        fill="magenta",
        label=STATION_NAME,
    )
    fig.legend(
        position="jTR+jTR+o0.5c",
        box="+gwhite+p1p",
    )

    # * 2. Plot the waveforms
    fig.shift_origin(yshift="-10i")
    fig.basemap(
        region=[
            0,
            40,
            min([item[2].min() for item in waveform_plot_array]) - WAVE_SCALE * 2,
            max([item[2].max() for item in waveform_plot_array]) + WAVE_SCALE * 2,
        ],
        projection="X3i/9.5i",
        frame=["WSen", 'xaf+l"Time (s)"', 'yaf+l"Hypocentral Distance (km)"'],
    )
    fig.text(
        x=2,
        y=max([item[2].max() for item in waveform_plot_array]) + WAVE_SCALE * 2 - 5,
        text="(b)",
        font="14p,Helvetica-Bold,black",
        justify="LM",
        no_clip=True,
    )
    for iwave, (event_id, color, wave, ps_time_diff, distance) in enumerate(
        waveform_plot_array
    ):
        fig.plot(
            x=np.linspace(0, 40, len(wave)),
            y=wave,
            pen=f"0.5p,{color}",
        )
        fig.plot(
            x=[10] * 2,
            y=[distance - WAVE_SCALE / 2, distance + WAVE_SCALE / 2],
            pen="1p,blue",
            label="P arrival" if iwave == 0 else None,
        )
        fig.plot(
            x=[10 + ps_time_diff] * 2,
            y=[distance - WAVE_SCALE / 2, distance + WAVE_SCALE / 2],
            pen="1p,black",
            label="PS arrival" if iwave == 0 else None,
        )
    fig.legend(
        position="jTR+jTR+o0.5c",
        box="+gwhite+p1p",
    )

    # * 3. Plot the beamforming figures
    XSHIFTS = [f"f{i+4.7}i" for i in [0, 2.8, 5.6]]
    YSHIFTS = [f"f{i+0.5}i" for i in [0, 2.5, 4.6, 7.1, 9.2, 11.7]]
    LABEL_NAMES = [
        "(e) PS beamforming after QC",
        "(d) PS beamforming before QC",
        "(c) P beamforming",
    ]
    FRAMES_bf = [
        [
            ["WSen", 'xaf+l"Azimuth Angle (degree)"', 'yaf+l"Takeoff Angle (degree)"'],
            ["WSen", 'xaf+l"Wave Speed (km/s)"', 'yaf+l"Takeoff Angle (degree)"'],
            ["WSen", 'xaf+l"Wave Speed (km/s)"', 'yaf+l"Azimuth Angle (degree)"'],
        ],
        [
            ["WSen", 'xaf+l"Wave Speed (km/s)"', 'yaf+l"Power"'],
            ["WSen", 'xaf+l"Azimuth Angle (degree)"', 'yaf+l"Power"'],
            ["WSen", 'xaf+l"Takeoff Angle (degree)"', 'yaf+l"Power"'],
        ],
    ]
    REGIONS_bf = [
        [
            [0, 360, -90, 90],
            [5.5, 11.5, -90, 90],
            [5.5, 11.5, 0, 360],
        ],
        [
            [5.5, 11.5, 0, 1.2],
            [0, 360, 0, 1.2],
            [-90, 90, 0, 1.2],
        ],
    ]
    pygmt.makecpt(cmap="jet", series=[0, 1, 0.1], continuous=True, reverse=False)
    for x in range(3):
        for y in range(6):
            fig.shift_origin(xshift=XSHIFTS[x], yshift=YSHIFTS[y])
            if y % 2 == 0:
                fig.basemap(
                    region=REGIONS_bf[y % 2][x],
                    projection="X2i/1.9i",
                    frame=FRAMES_bf[y % 2][x],
                )
                fig.grdimage(
                    images_2d[2 - y // 2][x],
                )
                if x == 2:
                    fig.shift_origin(xshift="f10.6i", yshift=YSHIFTS[y])
                    fig.colorbar(
                        frame=["x+lPower", "y"],
                        position="JMR+o0.2c",
                        scale=1,
                        box="+gwhite+p1p",
                    )
            else:
                fig.basemap(
                    region=REGIONS_bf[y % 2][x],
                    projection="X2i/1.1i",
                    frame=FRAMES_bf[y % 2][x],
                )
                fig.plot(
                    x=images_1d_x[2 - y // 2][x],
                    y=images_1d_y[2 - y // 2][x],
                    pen="1p",
                )
                # plot vertical dashed line for expected value
                fig.plot(
                    x=[images_1d_expected[2 - y // 2][x]] * 2,
                    y=[0, 1.2],
                    pen="1p,black,-",
                )
                if x == 0:
                    fig.text(
                        x=6,
                        y=1.4,
                        text=f"{LABEL_NAMES[y // 2]}",
                        font="14p,Helvetica-Bold,black",
                        justify="LM",
                        no_clip=True,
                    )

    save_path(fig, Path(__file__).resolve().stem)
