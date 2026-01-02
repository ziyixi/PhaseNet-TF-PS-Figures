"""
manualscript_ps_residuals.py:

This script is used to plot the PS arrivals residuals histograms for different stages of the semi-supervised learning.
"""

from pathlib import Path
from string import ascii_lowercase

import numpy as np
import pandas as pd
import pygmt

from phasenettfps import resource, save_path

TITLES = [
    "PhaseNet-TF@^(Iteration #1)",
    "PhaseNet-TF@^(Iteration #2)",
]


def main():
    fig = pygmt.Figure()
    pygmt.config(
        FONT_LABEL="auto",
        MAP_LABEL_OFFSET="12p",
        FONT_ANNOT_PRIMARY="12p",
        MAP_FRAME_TYPE="plain",
        MAP_TITLE_OFFSET="12p",
        FONT_TITLE="14p,black",
        MAP_FRAME_PEN="1p,black",
    )
    dfs = load_data()

    with fig.subplot(
        nrows=1,
        ncols=2,
        subsize=("2i", "2i"),
        margins=["0.1i", "0.05i"],
        frame=["WSen", "xaf", "yaf"],
        sharex="b",
        sharey="l",
    ):
        for i in range(2):
            with fig.set_panel(panel=i):
                fig.basemap(
                    projection="X?i/?i",
                    region=[-1.0, 1.0, 0, 800],
                    frame=[
                        f"+t{TITLES[i]}",
                        "xa0.4f+lTime Difference (s)",
                        "yaf+lCount",
                    ],
                )
                fig.histogram(
                    data=dfs[i]["residual"].values,
                    series=0.05,
                    pen="1p,black",
                    center=True,
                )
                diff_list = dfs[i]["residual"].values
                fig.text(
                    position="TR",
                    text=f"mean: {np.mean(diff_list):.2f} s",
                    font="14p,Helvetica,red",
                    offset="j0.15i/0.2i",
                )
                fig.text(
                    position="TR",
                    text=f"std: {np.std(diff_list):.2f} s",
                    font="14p,Helvetica,red",
                    offset="j0.2i/0.4i",
                )
                fig.text(
                    position="TL",
                    text=f"({ascii_lowercase[i]})",
                    font="14p,Helvetica-Bold,black",
                )
    save_path(fig, Path(__file__).resolve().stem)


# *============================== helper functions ==============================*
def load_data():
    ref_df = pd.read_csv(
        resource(["catalog", "tonga_picks_updated_2023_0426.csv"], normal_path=True)
    )
    first_round_df = pd.read_csv(
        resource(["catalog", "1st_round_inference.csv"], normal_path=True)
    )
    second_round_df = pd.read_csv(
        resource(["catalog", "2nd_round_inference.csv"], normal_path=True)
    )
    # construct dfs with columns station, time
    ref_df.dropna(subset=["arrival_time_PS"], inplace=True)
    ref_df = ref_df[["stacode", "arrival_time_PS"]]
    ref_df.rename(
        columns={"stacode": "station", "arrival_time_PS": "time"}, inplace=True
    )
    ref_df["time"] = pd.to_datetime(ref_df["time"])

    first_round_df = first_round_df[first_round_df["phase"] == "PS"]
    first_round_df = first_round_df[["sta", "time"]]
    first_round_df.rename(columns={"sta": "station", "time": "time"}, inplace=True)
    first_round_df["time"] = pd.to_datetime(first_round_df["time"]).dt.tz_localize(None)

    second_round_df = second_round_df[second_round_df["phase"] == "PS"]
    second_round_df = second_round_df[["sta", "time"]]
    second_round_df.rename(columns={"sta": "station", "time": "time"}, inplace=True)
    second_round_df["time"] = pd.to_datetime(
        second_round_df["time"], format="ISO8601"
    ).dt.tz_localize(None)

    all_stations_set = set(ref_df["station"].values)
    merged_1st = []
    merged_2nd = []
    for station in all_stations_set:
        ref = ref_df[ref_df["station"] == station]
        first = first_round_df[first_round_df["station"] == station]
        second = second_round_df[second_round_df["station"] == station]

        # merge asof for time
        # ref.sort_values(by="time", inplace=True)
        ref = ref.sort_values(by="time")
        first = first.sort_values(by="time")
        second = second.sort_values(by="time")
        first.rename(columns={"time": "time_1st"}, inplace=True)
        second.rename(columns={"time": "time_2nd"}, inplace=True)

        merged_ref_1st = pd.merge_asof(
            ref,
            first,
            left_on="time",
            right_on="time_1st",
            direction="nearest",
            tolerance=pd.Timedelta("1s"),
        )
        merged_ref_2nd = pd.merge_asof(
            ref,
            second,
            left_on="time",
            right_on="time_2nd",
            direction="nearest",
            tolerance=pd.Timedelta("1s"),
        )
        merged_1st.append(merged_ref_1st)
        merged_2nd.append(merged_ref_2nd)

    merged_1st = pd.concat(merged_1st)
    merged_2nd = pd.concat(merged_2nd)
    merged_1st.dropna(subset=["time_1st"], inplace=True)
    merged_2nd.dropna(subset=["time_2nd"], inplace=True)

    merged_1st["residual"] = (
        merged_1st["time_1st"] - merged_1st["time"]
    ).dt.total_seconds()
    merged_2nd["residual"] = (
        merged_2nd["time_2nd"] - merged_2nd["time"]
    ).dt.total_seconds()

    return merged_1st, merged_2nd
