"""
manuscript_final_map_compare.py:

This script is used to plot the PS arrivals count on the map, comparing the starting and final models.
"""

import json
from collections import defaultdict
from pathlib import Path
from string import ascii_lowercase

import numpy as np
import pandas as pd
import pygmt
import xarray as xr

from phasenettfps import resource, save_path

REGION = [-184, -172, -26, -14]


def main():
    fig = pygmt.Figure()
    pygmt.config(
        FONT_LABEL="auto",
        MAP_LABEL_OFFSET="auto",
        FONT_ANNOT_PRIMARY="auto",
        MAP_FRAME_TYPE="plain",
        MAP_TITLE_OFFSET="auto",
        FONT_TITLE="14p,black",
        MAP_FRAME_PEN="1p,black",
    )
    fig.shift_origin(yshift="1.5i")
    resource_dfs = [
        pd.read_csv(
            resource(
                ["catalog", "tonga_picks_updated_2023_0426.csv"], normal_path=True
            ),
        ),
        pd.read_csv(
            resource(["catalog", "first_round_result.csv"], normal_path=True),
            index_col=0,
        ),
    ]
    resource_dfs[1].rename(
        columns={
            "PSTIME": "arrival_time_PS",
            "STATION": "stacode",
            "EVENT_ID": "originid",
        },
        inplace=True,
    )
    LABELS = [
        "Number of manually picked PS arrivals",
        "Number of PS arrivals predicted by PhaseNet-TF (Iteration #1)",
    ]
    TEXTS = [
        "Reference",
        "PhaseNet-TF (Iteration #1)",
    ]

    with fig.subplot(
        nrows=2,
        ncols=2,
        subsize=("6i", "6i"),
        margins=["0.1i", "0.6i"],
        frame=["WSen", "xaf", "yaf"],
        sharex="b",
        sharey="l",
    ):
        for i in range(2):
            for j in range(2):
                with fig.set_panel(panel=(i, j)):
                    fig.basemap(
                        region=REGION,
                        projection="M?",
                        frame=["xaf", "yaf"],
                    )
                    plot_earth_relief(fig)
                    plot_text(fig)
                    if i == 0:
                        plot_stations(fig, resource_dfs[j], LABELS[j])
                    else:
                        plot_events(fig, resource_dfs[j], j)
                    plot_inset(fig)
                    plot_arrow(fig)
                    fig.text(
                        position="TL",
                        text=f"({ascii_lowercase[i*2+j]}) {TEXTS[j]}",
                        font="20p,Helvetica-Bold,black",
                    )

    save_path(fig, Path(__file__).resolve().stem)


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
        pen="1.6p,gray",
        limit="-10000/-7000",
    )
    # plot only -1000m contour
    fig.grdcontour(
        grd_topo,
        interval=1000,
        pen="1p,gray",
        limit="-1100/-1000",
    )
    fig.coast(land="gray")


def plot_text(fig: pygmt.Figure):
    text_elements = [
        {
            "text": "Tonga Trench",
            "x": -173.3,
            "y": -22.5,
            "font": "18p,Helvetica-Bold,black",
            "angle": 65,
        },
        {
            "text": "Lau Basin",
            "x": -176,
            "y": -18,
            "font": "18p,Helvetica-Bold,black",
            "angle": 65,
        },
        {
            "text": "Fiji",
            "x": 178,
            "y": -17,
            "font": "18p,Helvetica-Bold,black",
            "angle": 0,
        },
    ]

    for element in text_elements:
        fig.text(**element)


def plot_stations(fig: pygmt.Figure, resource_dfs: pd.DataFrame, label_text=""):
    with open(resource(["stations", "stations.json"], normal_path=True)) as f:
        stations = json.load(f)
    # json's key is station name, value is a dict, with keys longitude, latitude, and local_depth_m
    # prepare a pandas dataframe for plotting
    station_df = []
    for station in stations:
        net, sta, _, _ = station.split(".")
        station_df.append(
            [
                net,
                sta,
                stations[station]["longitude"],
                stations[station]["latitude"],
            ]
        )
    station_df = pd.DataFrame(station_df, columns=["net", "sta", "lon", "lat"])
    # now count how many PS arrivals for each station
    df_stations_catalog = resource_dfs
    counter = defaultdict(int)
    for _, row in df_stations_catalog.iterrows():
        if not pd.isnull(row["arrival_time_PS"]):
            counter[row["stacode"]] += 1
    # add the counter to the station_df
    station_df["count"] = station_df["sta"].map(counter)
    # make count==0 as np.nan
    station_df["count"] = station_df["count"].replace(0, np.nan)

    min_count = np.nanmin(station_df["count"].to_list())
    max_count = np.nanmax(station_df["count"].to_list())
    pygmt.makecpt(
        cmap="gray",
        series=[min_count, max_count, 1],
        continuous=True,
        reverse=True,
    )

    fig.plot(
        x=station_df[(station_df["net"] == "YL") & (pd.isnull(station_df["count"]))][
            "lon"
        ],
        y=station_df[(station_df["net"] == "YL") & (pd.isnull(station_df["count"]))][
            "lat"
        ],
        style="i0.3c",
        pen="0.5p,black",
        label="YL",
        fill="white",
    )
    fig.plot(
        x=station_df[(station_df["net"] == "Z1") & (pd.isnull(station_df["count"]))][
            "lon"
        ],
        y=station_df[(station_df["net"] == "Z1") & (pd.isnull(station_df["count"]))][
            "lat"
        ],
        style="t0.3c",
        pen="0.5p,black",
        label="Z1",
        fill="white",
    )
    fig.plot(
        x=station_df[(station_df["net"] == "II") & (pd.isnull(station_df["count"]))][
            "lon"
        ],
        y=station_df[(station_df["net"] == "II") & (pd.isnull(station_df["count"]))][
            "lat"
        ],
        style="s0.3c",
        pen="0.5p,black",
        label="II",
        fill="white",
    )
    fig.legend(
        position="JBR+jBR+o0.3c/0.2c+w1.2c/1.5c",
        box="+gwhite+p1p,white",
    )

    # plot net==YL as reverse triangle, net==Z1 as triange, II as diamond
    fig.plot(
        x=station_df[(station_df["net"] == "YL") & (~pd.isnull(station_df["count"]))][
            "lon"
        ],
        y=station_df[(station_df["net"] == "YL") & (~pd.isnull(station_df["count"]))][
            "lat"
        ],
        style="i0.3c",
        pen="0.5p,black",
        # label="YL",
        fill=station_df[
            (station_df["net"] == "YL") & (~pd.isnull(station_df["count"]))
        ]["count"],
        cmap=True,
    )
    fig.plot(
        x=station_df[(station_df["net"] == "Z1") & (~pd.isnull(station_df["count"]))][
            "lon"
        ],
        y=station_df[(station_df["net"] == "Z1") & (~pd.isnull(station_df["count"]))][
            "lat"
        ],
        style="t0.3c",
        pen="0.5p,black",
        # label="Z1",
        fill=station_df[
            (station_df["net"] == "Z1") & (~pd.isnull(station_df["count"]))
        ]["count"],
        cmap=True,
    )
    # ! no colored II stations
    # fig.plot(
    #     x=station_df[(station_df["net"] == "II") & (~pd.isnull(station_df["count"]))][
    #         "lon"
    #     ],
    #     y=station_df[(station_df["net"] == "II") & (~pd.isnull(station_df["count"]))][
    #         "lat"
    #     ],
    #     style="s0.3c",
    #     pen="0.5p,black",
    #     # label="II",
    #     fill=station_df[
    #         (station_df["net"] == "II") & (~pd.isnull(station_df["count"]))
    #     ]["count"],
    #     cmap=True,
    # )
    # plot legend in the bottom right corner
    # fig.legend(
    #     position="JBR+jBR+o0.2c/0.2c",
    #     box="+gwhite+p1p,black",
    # )

    fig.colorbar(
        position="JMB+o0c/0.8c+w5i/0.3i",
        frame=[f"x+l{label_text}"],
    )


def plot_inset(fig: pygmt.Figure):
    with fig.inset(position="jBL+w7c+o0.2c", margin=0):
        fig.coast(
            region="g",
            projection=f"W{(REGION[0]+REGION[1])//2}/6c",
            land="gray",
            water="white",
            frame=["wsen", "xafg", "yafg"],
        )
        fig.plot(
            x=[REGION[0], REGION[1], REGION[1], REGION[0], REGION[0]],
            y=[REGION[2], REGION[2], REGION[3], REGION[3], REGION[2]],
            pen="1p,black",
            projection=f"W{(REGION[0]+REGION[1])//2}/6c",
        )


def plot_events(fig: pygmt.Figure, resource_dfs: pd.DataFrame, j):
    if j == 0:
        df_events = pd.read_csv(
            resource(
                ["catalog", "tonga_catalog_updated_2023_0426.csv"], normal_path=True
            ),
            parse_dates=["time"],
        )
    else:
        df_events = pd.read_csv(
            resource(
                ["catalog", "deep_learning_for_deep_earthquakes_catalog.csv"],
                normal_path=True,
            ),
            parse_dates=["time"],
        )
        # rename id to originid
        df_events.rename(columns={"id": "originid"}, inplace=True)
    # for df_events, counts the number of PS arrivals for each event
    counter = defaultdict(int)
    for _, row in resource_dfs.iterrows():
        if not pd.isnull(row["arrival_time_PS"]):
            counter[row["originid"]] += 1
    df_events["count"] = df_events["originid"].map(counter)
    # normalize count to [0.05,0.4]
    min_count = df_events["count"].min()
    max_count = df_events["count"].max()
    df_events["count"] = df_events["count"].apply(
        lambda x: 0.05 + 0.45 * (x - min_count) / (max_count - min_count)
    )

    pygmt.makecpt(cmap="jet", series=[0, 700, 1], continuous=True, reverse=True)
    fig.plot(
        x=df_events["longitude"],
        y=df_events["latitude"],
        size=df_events["count"],
        style="c",
        fill=df_events["depth"],
        cmap=True,
    )
    # plot the colorbar to the right of the map
    fig.colorbar(
        position="JMB+o0c/0.8c+w5i/0.3i",
        frame=["x+lEarthquake depth (km)"],
    )


def plot_plate_boundary(fig: pygmt.Figure):
    x_lists, y_lists = load_bord_plate_boundaries()
    for x, y in zip(x_lists, y_lists):
        fig.plot(
            x=x,
            y=y,
            pen="3p,magenta",
        )


def load_bord_plate_boundaries():
    """
    Load the plate boundaries from the bord's plate boundaries file
    """
    x_lists = [[]]
    y_lists = [[]]
    with open(
        resource(["Plate_Boundaries", "bird_2002_boundaries"], normal_path=True)
    ) as f:
        for line in f.readlines():
            if line.startswith(" "):
                x_lists[-1].append(float(line.split(",")[0]))
                y_lists[-1].append(float(line.split(",")[1]))
            elif line.startswith("*"):
                x_lists.append([])
                y_lists.append([])
    x_lists = [x for x in x_lists if len(x) > 1]
    y_lists = [y for y in y_lists if len(y) > 1]
    return x_lists, y_lists


def plot_arrow(fig):
    style = "V1c+e+h0"
    fig.plot(
        x=[-173.5],
        y=[-25],
        style=style,
        direction=([-65], [2]),
        pen="4p,black",
        fill="black",
    )
