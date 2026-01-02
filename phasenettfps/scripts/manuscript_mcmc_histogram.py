"""
manuscript_mcmc_histogram.py:

This script is used to generate the histogram of the MCMC trustworthiness scores for the manuscript.
"""

import pickle
from collections import defaultdict
from pathlib import Path
from string import ascii_lowercase

import arviz as az
import numpy as np
import pygmt

from phasenettfps import resource, save_path


def load_data():
    # * load manual MCMC
    MCMC_DIR_manual = Path(
        resource(
            ["mcmc", "mcmcbf_manual_filtered__v_takeoff_1_3_15_45"], normal_path=True
        )
    )
    mcmc_res_manual = defaultdict(list)
    for file in MCMC_DIR_manual.glob("*.nc"):
        trace = az.from_netcdf(file)
        df = az.summary(trace, round_to=2)
        with open(
            MCMC_DIR_manual / f"idx2trueidx_{file.stem.split('_')[-1]}.pkl", "rb"
        ) as f:
            mapper = pickle.load(f)

        for i, row in df.iterrows():
            mcmc_res_manual[mapper[int(i[2:-1])]].append(row["mean"])
    mcmc_mean_manual = []
    for k, v in mcmc_res_manual.items():
        mcmc_mean_manual.append(np.mean(v))

    # * load first iteration MCMC
    MCMC_DIR_first = Path(
        resource(["mcmc", "mcmcbf_first_round_prediction"], normal_path=True)
    )
    mcmc_res_first = defaultdict(list)
    for file in MCMC_DIR_first.glob("*.nc"):
        trace = az.from_netcdf(file)
        df = az.summary(trace, round_to=2)
        with open(
            MCMC_DIR_first / f"idx2trueidx_{file.stem.split('_')[-1]}.pkl", "rb"
        ) as f:
            mapper = pickle.load(f)

        for i, row in df.iterrows():
            mcmc_res_first[mapper[int(i[2:-1])]].append(row["mean"])
    mcmc_mean_first = []
    for k, v in mcmc_res_first.items():
        mcmc_mean_first.append(np.mean(v))

    return {
        "manual": mcmc_mean_manual,
        "first round PhaseNet-TF": mcmc_mean_first,
    }


def main():
    info = load_data()

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

    panel_names = [
        "manual",
        "first round PhaseNet-TF",
        "first round PhaseNet-TF",
    ]
    max_height = [1000, 4000, 4000]
    with fig.subplot(
        nrows=1,
        ncols=3,
        figsize=("9.6i", "3i"),
        sharex=True,
        sharey=True,
        margins="0.3i",
    ):
        for i in range(2):
            with fig.set_panel(i):
                fig.basemap(
                    region=[0, 1, 0, max_height[i]],
                    projection="X?/?",
                    frame=["WSen", 'x0.2f+l"Trustworthiness Expection"', 'yaf+l"Count"']
                    if i == 0
                    else ["WSen", 'x0.2f+l"Trustworthiness Expection"', "yaf"],
                )
                fig.histogram(
                    data=info[panel_names[i]],
                    series=0.05,
                    pen="2p,black",
                )
                # label at top left
                fig.text(
                    position="TL",
                    text=f"({ascii_lowercase[i]})",
                    font="18p,black",
                    offset="4p/-4p",
                )
                fig.text(
                    position="TR",
                    text=panel_names[i],
                    font="14p,red",
                    offset="-4p/-4p",
                )

    save_path(fig, Path(__file__).resolve().stem)
