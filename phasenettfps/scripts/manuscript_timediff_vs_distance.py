"""
manuscript_timediff_vs_distance.py:

This script is used to generate the time difference vs distance figure for the manuscript.
"""

import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt
from numba import njit
from tqdm import tqdm

from phasenettfps import resource, save_path


@njit
def lat_lon_dep_to_cartesian(lat, lon, dep):
    """
    Convert latitude, longitude, and depth into Cartesian coordinates.
    Latitude and Longitude in degrees, depth in kilometers.
    """
    # Earth's radius in kilometers (average)
    R = 6371.0
    # Convert latitude and longitude from degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Calculate Cartesian coordinates
    r = R - dep  # Adjust radius for depth
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return x, y, z


@njit
def kernel(event, station, event_lat, event_lon, event_dep, station_lat, station_lon):
    """
    Calculate the hypocentral distance between an earthquake event and a station.
    Coordinates in degrees, depth in kilometers.
    """
    # Convert earthquake and station positions to Cartesian coordinates
    eq_x, eq_y, eq_z = lat_lon_dep_to_cartesian(event_lat, event_lon, event_dep)
    st_x, st_y, st_z = lat_lon_dep_to_cartesian(
        station_lat, station_lon, 0
    )  # Station depth = 0

    # Calculate the 3D distance
    distance = np.sqrt((eq_x - st_x) ** 2 + (eq_y - st_y) ** 2 + (eq_z - st_z) ** 2)

    return event, station, distance


def kernel_batch(batch, events_coordinates, stations_coordinates):
    results = []
    for event, station in batch:
        result = kernel(
            event,
            station,
            events_coordinates[event]["lat"],
            events_coordinates[event]["lon"],
            events_coordinates[event]["dep"],
            stations_coordinates[station]["lat"],
            stations_coordinates[station]["lon"],
        )
        results.append(result)
    return results


def create_batches(all_pairs, batch_size):
    """Yield successive n-sized chunks from all_pairs."""
    for i in range(0, len(all_pairs), batch_size):
        yield all_pairs[i : i + batch_size]


def load_time_diff_vs_distance():
    # * load manual picks
    manual_catalog = pd.read_csv(
        resource(["catalog", "tonga_catalog_updated_2023_0426.csv"], normal_path=True)
    )
    manual_picks = pd.read_csv(
        resource(["catalog", "tonga_picks_updated_2023_0426.csv"], normal_path=True)
    )
    # remove stacode==F01
    manual_picks = manual_picks[manual_picks["stacode"] != "F01"]

    with open(resource(["catalog", "stations.json"], normal_path=True), "r") as f:
        stations = json.load(f)

    all_station_eventid_pairs = set()
    manual_picks.apply(
        lambda x: all_station_eventid_pairs.add((x["originid"], x["stacode"])), axis=1
    )
    stations_coordinates = {
        k.split(".")[1]: {"lat": v["latitude"], "lon": v["longitude"]}
        for k, v in stations.items()
    }
    events_coordinates = {}
    manual_catalog.apply(
        lambda x: events_coordinates.update(
            {
                x["originid"]: {
                    "lat": x["latitude"],
                    "lon": x["longitude"],
                    "dep": x["depth"],
                }
            }
        ),
        axis=1,
    )

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for event, station in all_station_eventid_pairs:
            futures.append(
                executor.submit(
                    kernel,
                    event,
                    station,
                    events_coordinates[event]["lat"],
                    events_coordinates[event]["lon"],
                    events_coordinates[event]["dep"],
                    stations_coordinates[station]["lat"],
                    stations_coordinates[station]["lon"],
                )
            )
    result = {}
    for each in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(futures),
        desc="Calculating distance for manual picks",
    ):
        event, station, distance = each.result()
        result[(event, station)] = distance

    # update picks df
    manual_picks["distance"] = manual_picks.apply(
        lambda x: result[(x["originid"], x["stacode"])], axis=1
    )
    # update arrival_time_S-arrival_time_P and arrival_time_PS-arrival_time_P
    manual_picks["arrival_time_S"] = pd.to_datetime(manual_picks["arrival_time_S"])
    manual_picks["arrival_time_P"] = pd.to_datetime(manual_picks["arrival_time_P"])
    manual_picks["arrival_time_PS"] = pd.to_datetime(manual_picks["arrival_time_PS"])
    manual_picks["PS-P"] = (
        manual_picks["arrival_time_PS"] - manual_picks["arrival_time_P"]
    ).dt.total_seconds()
    manual_picks["S-P"] = (
        manual_picks["arrival_time_S"] - manual_picks["arrival_time_P"]
    ).dt.total_seconds()

    # * load first iteration result
    def load_iteration_related(df, load_s=True):
        df.dropna(subset=["PSTIME"], inplace=True)

        all_station_eventid_pairs = set()
        if "NET_STA" in df.columns:
            df.apply(
                lambda x: all_station_eventid_pairs.add(
                    (x["EVENT_ID"], x["NET_STA"].split(".")[1])
                ),
                axis=1,
            )
        else:
            df.apply(
                lambda x: all_station_eventid_pairs.add((x["EVENT_ID"], x["STATION"])),
                axis=1,
            )

        events_coordinates = {}
        stations_coordinates = {}

        def update_kernel(row):
            events_coordinates.update(
                {
                    row["EVENT_ID"]: {
                        "lat": row["ELAT"],
                        "lon": row["ELON"],
                        "dep": row["EDEP"],
                    }
                }
            )
            if "NET_STA" in df.columns:
                stations_coordinates.update(
                    {
                        row["NET_STA"].split(".")[1]: {
                            "lat": row["SLAT"],
                            "lon": row["SLON"],
                        }
                    }
                )
            else:
                stations_coordinates.update(
                    {row["STATION"]: {"lat": row["SLAT"], "lon": row["SLON"]}}
                )

        df.apply(update_kernel, axis=1)

        batch_size = 100  # Define the size of each batch
        batches = create_batches(list(all_station_eventid_pairs), batch_size)

        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    kernel_batch, batch, events_coordinates, stations_coordinates
                )
                for batch in batches
            ]

        results_futures = []
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Calculating distance for first iteration result",
        ):
            results_futures.extend(future.result())

        result = {}
        for each in results_futures:
            event, station, distance = each
            result[(event, station)] = distance

        # update first iteration result df
        if "NET_STA" in df.columns:
            df["distance"] = df.apply(
                lambda x: result[(x["EVENT_ID"], x["NET_STA"].split(".")[1])], axis=1
            )
        else:
            df["distance"] = df.apply(
                lambda x: result[(x["EVENT_ID"], x["STATION"])], axis=1
            )
        # update arrival_time_S-arrival_time_P and arrival_time_PS-arrival_time_P
        df["PTIME"] = pd.to_datetime(df["PTIME"])
        if load_s:
            df["STIME"] = pd.to_datetime(df["STIME"])
            df["S-P"] = (df["STIME"] - df["PTIME"]).dt.total_seconds()
        df["PSTIME"] = pd.to_datetime(df["PSTIME"])
        df["PS-P"] = (df["PSTIME"] - df["PTIME"]).dt.total_seconds()

        return df

    # * load first iteration result
    first_iteration_result = pd.read_csv(
        resource(["catalog", "first_round_result.csv"], normal_path=True),
        index_col=0,
        parse_dates=["PSTIME"],
    )
    first_iteration_result = load_iteration_related(first_iteration_result, load_s=True)

    # * load before energy
    before_energy = pd.read_csv(
        resource(["catalog", "1st_round_before_energy.csv"], normal_path=True),
        parse_dates=["PSTIME"],
    )
    before_energy = load_iteration_related(before_energy, load_s=True)

    # * load before bf
    before_bf = pd.read_csv(
        resource(["catalog", "1st_round_for_beamforming.csv"], normal_path=True),
        parse_dates=["PSTIME"],
    )
    before_bf = load_iteration_related(before_bf, load_s=False)

    # fit first_iteration_result["S-P"] to a line from the origin
    first_iteration_result_toxy = first_iteration_result.dropna(
        subset=["distance", "S-P"]
    )
    x = np.array(first_iteration_result_toxy["distance"].to_list())
    y = np.array(first_iteration_result_toxy["S-P"].to_list())

    @njit
    def calculate_ratio_above(x, y, slope):
        y_pred = x * slope
        count = np.sum(y > y_pred)
        return count / len(y)

    def get_slopes(x, y):
        test_slopes = np.arange(0, 0.5, 0.001)
        ratios = [calculate_ratio_above(x, y, slope) for slope in test_slopes]
        # find the slope that is nearest to 0.05,0.5,0.95 for ratio
        upper_slope = test_slopes[np.argmin(np.abs(np.array(ratios) - 0.05))]
        slope = test_slopes[np.argmin(np.abs(np.array(ratios) - 0.5))]
        lower_slope = test_slopes[np.argmin(np.abs(np.array(ratios) - 0.95))]
        return slope, lower_slope, upper_slope

    slope, lower_slope, upper_slope = get_slopes(x, y)
    print(slope, lower_slope, upper_slope)

    def filter_func(row):
        # get all rows that is above lower_slope bound
        return row["PS-P"] > row["distance"] * lower_slope

    removed_due_to_slope = before_energy[before_energy.apply(filter_func, axis=1)]
    before_energy = before_energy.drop(removed_due_to_slope.index)

    removed_base_on_time = before_energy[before_energy["PS-P"] < 5]
    before_energy = before_energy.drop(removed_base_on_time.index)

    # filter first_iteration_result based on the slope
    first_iteration_result_remove = first_iteration_result[
        first_iteration_result.apply(filter_func, axis=1)
    ]
    first_iteration_result = first_iteration_result.drop(
        first_iteration_result_remove.index
    )

    return (
        manual_picks,
        first_iteration_result,
        before_energy,
        before_bf,
        removed_base_on_time,
        removed_due_to_slope,
        (slope, lower_slope, upper_slope),
    )


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

    (
        df_start,
        df_final,
        df_final_before_energy,
        df_final_before_bf,
        removed_base_on_time,
        removed_due_to_slope,
        (slope, lower_slope, upper_slope),
    ) = load_time_diff_vs_distance()

    def random_sample(df, ratio=1 / 3):
        return df.sample(frac=ratio, random_state=1)

    df_start = random_sample(df_start)
    df_final = random_sample(df_final)
    df_final_before_energy = random_sample(df_final_before_energy)
    df_final_before_bf = random_sample(df_final_before_bf)
    removed_base_on_time = random_sample(removed_base_on_time)

    # for df_start, only keep rows where PS-P are not NaN
    df_start = df_start.dropna(subset=["PS-P"])
    df_final = df_final.dropna(subset=["PS-P"])

    # remove all rows where S-P is not nan and S-P <= PS-P
    df_start = df_start.drop(
        df_start[
            (df_start["S-P"].notna()) & (df_start["S-P"] <= df_start["PS-P"])
        ].index
    )
    df_final = df_final.drop(
        df_final[
            (df_final["S-P"].notna()) & (df_final["S-P"] <= df_final["PS-P"])
        ].index
    )

    dfs = [df_start, df_final]
    dfs_other = [None, [df_final_before_energy, df_final_before_bf]]
    TITLES = ["Manual", "Final"]

    with fig.subplot(
        nrows=1,
        ncols=2,
        figsize=("10.1i", "5i"),
        sharex="b",
        sharey="l",
        margins=["0.1i", "0.06i"],
        frame=[
            "WSen",
        ],
    ):
        for i in range(2):
            with fig.set_panel(i):
                fig.basemap(
                    projection="X?/Y?",
                    region=[0, 1200, 0, 100],
                    frame=[
                        'xaf+l"Hypocentral distance (km)"',
                        'yaf+l"Differencial traveltime (sec)"',
                        f"+t{TITLES[i]}",
                    ],
                )
                # plot blue + for S-P
                fig.plot(
                    x=dfs[i]["distance"],
                    y=dfs[i]["S-P"],
                    style="+0.2c",
                    fill="magenta",
                    pen="magenta",
                    label="S-P",
                )

                # * plot other
                if dfs_other[i] is not None:
                    # * second remove, energy constrain
                    fig.plot(
                        x=dfs_other[i][0]["distance"],
                        y=dfs_other[i][0]["PS-P"],
                        style="c0.08c",
                        fill="green",
                        pen="0.5p,green",
                        label="PS-P removed due to energy",
                    )
                    # * third remove, mcmc constrain
                    fig.plot(
                        x=dfs_other[i][1]["distance"],
                        y=dfs_other[i][1]["PS-P"],
                        style="c0.08c",
                        fill="blue",
                        pen="blue",
                        label="PS-P removed due to beamforming",
                    )
                    # * first remove, 5s constrain
                    fig.plot(
                        x=removed_base_on_time["distance"],
                        y=removed_base_on_time["PS-P"],
                        style="c0.08c",
                        fill="black",
                        pen="black",
                        label="PS-P removed due to travel time",
                    )
                # plot red circle for PS-P
                fig.plot(
                    x=dfs[i]["distance"],
                    y=dfs[i]["PS-P"],
                    style="c0.1c",
                    # fill="white",
                    pen="1p,red",
                    label="Final PS-P",
                    transparency=90,
                )
                if dfs_other[i] is not None:
                    # * plot removed_due_to_slope
                    fig.plot(
                        x=removed_due_to_slope["distance"],
                        y=removed_due_to_slope["PS-P"],
                        style="c0.08c",
                        fill="black",
                        pen="black",
                    )
                # plot the linear fit
                fig.plot(
                    x=[0, 1200],
                    y=[0, 1200 * slope],
                    pen="1p,black",
                )
                # plot the 95% confidence interval
                fig.plot(
                    x=[0, 1200],
                    y=[0, 1200 * lower_slope],
                    pen="1p,black,-.",
                )
                fig.plot(
                    x=[0, 1200],
                    y=[0, 1200 * upper_slope],
                    pen="1p,black,-.",
                )

                fig.legend(
                    position="JTL+jTL+o0.2c/0c",
                    box="+gwhite+p1p",
                )
                # text on the top right corner for the number of PS arrivals
                fig.text(
                    position="TR",
                    text=f"Number of selected PS arrivals: {len(dfs[i])*3}",
                )

    save_path(fig, Path(__file__).resolve().stem)
