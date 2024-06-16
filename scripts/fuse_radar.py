from itertools import combinations
import os
from pathlib import Path
import sys
import click
import polars as pl
import dotenv
import geopandas as gpd

import numpy as np
from scipy.stats import chi2
import networkx as nx

import functools

from tqdm import tqdm

ROOT = Path(os.getcwd())
while not ROOT.joinpath(".git").exists():
    ROOT = ROOT.parent
# add the root to the python path
sys.path.append(str(ROOT))

dotenv.load_dotenv(ROOT / ".env")
from src.filters.fusion import batch_join, rts_smooth  # noqa: E402
from src.geometry import RoadNetwork  # noqa: E402
from src.utils import check_gpu_available  # noqa: E402
from src.pipelines.kalman_filter import (  # noqa: E402
    filter_short_trajectories,
    join_results,
    prepare_frenet_measurement,
    build_extension,
    add_timedelta,
    build_kalman_id,
    build_kalman_df,
)
from src.filters.vectorized_kalman import batch_imm_df  # noqa: E402
from src.pipelines.association import (  # noqa: E402
    add_front_back_s,
    build_fusion_df,
    build_leader_follower_entire_history_df,
    build_leader_follower_no_sort,  # noqa: F401
    calc_assoc_liklihood_distance,
    create_vehicle_ids,
    make_graph_based_ids,
    pipe_gate_headway_calc,
    build_match_df as build_match_pipeline,
    calculate_match_indexes,
)
from src.pipelines.graph import walk_graph_removals  # noqa: E402
from src.pipelines.lane_classification import label_lanes_tree  # noqa: E402
from src.radar import CalibratedRadar  # noqa: E402

# from src.filters.lowpass import butter_lowpass_filter_plot  # noqa: E402

GPU = check_gpu_available()

IMM_M = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
IMM_MU = np.array([0.1, 0.1, 0.8])

CACHE = True


def _list_converter(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col(df.columns[i]).arr.to_list()
        for i, dtype in enumerate(df.dtypes)
        if "Array" in str(dtype)
    )


def _array_converter(df: pl.DataFrame):
    df = df.lazy()
    list_cols = []
    for col, dtype in zip(df.columns, df.dtypes):
        if "List" in str(dtype):
            list_cols.append(col)

    lens = df.select(pl.col(list_cols).list.len().first()).fetch(1)

    return df.with_columns(
        pl.col(col).list.to_array(width=lens[col][0]) for col in list_cols
    )


def cache_wrapper(output_name_arg):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if CACHE:
                cache_dir = os.environ.get("CACHE_DIR")
                if cache_dir is None:
                    raise ValueError("Environment variable 'CACHE_DIR' is not set.")

                output_name = output_name_arg

                output_file = f"{func.__name__}_{output_name}.parquet"
                output_path = os.path.join(cache_dir, output_file)

                if os.path.exists(output_path):
                    return pl.scan_parquet(output_path).pipe(_array_converter).collect()

                result = func(*args, **kwargs)
                assert isinstance(result, pl.DataFrame)
                result.lazy().pipe(_list_converter).collect().write_parquet(output_path)

                return result
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def create_helper_objs(calibration_yaml: str) -> tuple:
    """
    Create the helper objects for the pipeline. This includes the mainline and full road network objects, and the radar object.

    Args:
        calibration_yaml (str): The path to the calibration yaml file, which specifies the radar location and rotation angles

    Returns:
        tuple: A tuple containing the mainline network, full network, and radar object
    """
    mainline_net = RoadNetwork(
        lane_gdf=gpd.read_file(ROOT / "data/mainline_lanes.geojson"),
        keep_lanes=["EBL1", "WBL1"],
        step_size=0.01,
    )

    full_net = RoadNetwork(
        lane_gdf=gpd.read_file(ROOT / "data/mainline_lanes.geojson"),
        keep_lanes=None,
        step_size=0.01,
    )

    radar_obj = CalibratedRadar(
        radar_location_path=calibration_yaml,
    )
    return mainline_net, full_net, radar_obj


# def _sql_filter(df: pl.LazyFrame, sql: str = None) -> pl.LazyFrame:

#     if sql is not None:
#         import duckdb

#         # duckdb works a lot better than polars with SQL queries
#         return duckdb.sql(
#             """
#         SELECT * FROM df WHERE {}
#         """.format(
#                 sql
#             )
#         ).pl().lazy()

#     return df


def _build_small_df(df: pl.LazyFrame, build_small: bool) -> pl.LazyFrame:
    if build_small:
        df = df.filter(
            (pl.col("epoch_time").dt.day() == 13)
            & (pl.col("epoch_time").dt.hour() == 12)
        )

        # save the small raw radar data. This is harcoded path b.c. lazy
        df.collect().write_parquet(
            "tmp/raw_radar_small.parquet", 
            use_pyarrow=True,
            compression_level=22
        )

    return df


@cache_wrapper("raw_radar_df")
def open_radar_data(
    path: str,
    radar_obj: CalibratedRadar,
    mainline_net: RoadNetwork,
    build_small_df: str = None,
) -> pl.DataFrame:
    """
    Open the radar data and perform the necessary preprocessing steps to prepare the data for the pipeline.

    Args:
        path (str): The path to the radar data. Can be a single file or a glob pattern
        radar_obj (CalibratedRadar): The radar object, which contains the calibration information
        mainline_net (RoadNetwork): The mainline network object, which contains the lane information
        sql (str, optional): A SQL filter to apply to the data. Defaults to None.

    Returns:
        pl.DataFrame: The preprocessed radar data
    """

    def dist_filter(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            pl.when(
                pl.col("ip").is_in(["10.160.7.141", "10.160.7.137"])
                & (pl.col("epoch_time").dt.month() == 10)
            )
            .then(pl.col("dist") < 275)
            .otherwise(pl.lit(True))
        )

    return (
        pl.scan_parquet(path)
        # .with_columns(
        #     pl.col("epoch_time").dt.replace_time_zone("UTC").dt.round("100ms"),
        # )
        .pipe(radar_obj.create_object_id)
        .pipe(_build_small_df, build_small=build_small_df)
        # sort by object_id and epoch_time
        .sort(by=["object_id", "epoch_time"])
        .set_sorted(["object_id", "epoch_time"])
        # filter out vehicles that don't trave some minimum distance (takes care of radar noise)
        # This happens when the radar tracks things like the opposite signal pole or signs
        .pipe(
            radar_obj.filter_short_trajectories,
            minimum_distance_m=2,
            minimum_duration_s=1,
        )
        # clip the end of trajectories where the velocity is constant
        .pipe(radar_obj.clip_trajectory_end)
        .pipe(radar_obj.resample, 100)
        .pipe(radar_obj.set_timezone, timezone_="UTC")
        # .pipe(radar_obj.add_cst_timezone)
        .pipe(radar_obj.add_heading)
        .pipe(radar_obj.rotate_radars)
        .pipe(radar_obj.update_origin)
        .with_columns(
            (pl.col("f32_positionX_m") ** 2 + pl.col("f32_positionY_m") ** 2)
            .sqrt()
            .alias("dist"),
        )
        # filter out the radar data that is far away from the problem radar heads
        .pipe(dist_filter)
        # mark whethter the vehicle is approaching the radar head or not
        .with_columns(
            (pl.col("dist").diff().backward_fill(1).over("object_id") < 0).alias(
                "approaching"
            )
        )
        .pipe(lambda df: df.with_row_count() if "row_nr" not in df.columns else df)
        .sort(by=["epoch_time"])
        .set_sorted(["epoch_time"])
        .collect(streaming=True)
        # map to the lanes
        .pipe(
            mainline_net.map_to_lane,
            dist_upper_bound=mainline_net.LANE_WIDTH * mainline_net.LANE_NUM
            - (mainline_net.LANE_WIDTH / 2)
            + 0.5,  # centered on one of the lanes,
            utm_x_col="utm_x",
            utm_y_col="utm_y",
        )
        .rename(
            {
                "name": "lane",
                "angle": "heading_lane",
            }
        )
    )


@cache_wrapper("radar_df")
def build_radar_df(
    raw_radar_df: pl.DataFrame,
    prediction_length: float,
    minimum_distance_m: float = 5,
    minimum_duration_s: float = 2,
) -> pl.DataFrame:
    """
    Function to pre-process the radar dataframe for the IMM filtering.

    Args:
        raw_radar_df (pl.DataFrame): The raw radar dataframe. Should be the output of the `open_radar_data` function
        prediction_length (float): The prediction length in seconds
        minimum_distance_m (float, optional): The minimum distance that a trajectory has to travel to be considered valid. Defaults to 5m.
        minimum_duration_s (float, optional): The minimum duration that a trajectory has to last to be considered valid. Defaults to 2s.

    Returns:
        pl.DataFrame: The preprocessed radar dataframe, ready for the IMM filtering
    """

    return (
        raw_radar_df.pipe(add_timedelta, vehicle_id_col=["object_id", "lane"])
        .pipe(build_kalman_id, split_time_delta=prediction_length + 0.1)
        .pipe(
            filter_short_trajectories,
            minimum_distance_m=minimum_distance_m,
            minimum_duration_s=minimum_duration_s,
        )
        .pipe(prepare_frenet_measurement)
        .pipe(build_extension, seconds=prediction_length)
        .pipe(add_timedelta)
        .collect()
    )


@cache_wrapper("joined_df")
def imm_filter(
    radar_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Function to apply the IMM filter to the radar dataframe

    Args:
        radar_df (pl.DataFrame): The radar dataframe. Should be the output of the `build_radar_df` function

    Returns:
        pl.DataFrame: The filtered radar dataframe
    """
    filter_df = (
        radar_df.pipe(build_kalman_df, s_col="s", derive_s_vel=False)
        .filter(
            pl.col("max_time")
            < pl.col("max_time").max()  # filter out the outlier times. This is a hack
        )
        .collect()
    )

    filt_df = batch_imm_df(
        filter_df.rename({"measurement": "z"}),
        filters=("CALC", "CALK", "CVLK"),
        M=IMM_M,
        mu=IMM_MU,
        chunk_size=100_000 if not GPU else 9_000 * 1000,
        gpu=GPU,
    )

    return (
        join_results(filt_df, filter_df, radar_df)
        .lazy()
        .pipe(
            add_front_back_s,
            use_global_mean=True,
        )
        .sort("epoch_time")
        .set_sorted(["epoch_time"])
        .collect(streaming=True)
        .rechunk()
    )


# @cache_wrapper("lf_df")
def build_lf_df(
    joined_df: pl.DataFrame,
    mainline_net: RoadNetwork,
    full_net: RoadNetwork,
) -> pl.DataFrame:
    """
    This function constructs the leader-follower dataframe. Vehices must have been leaders and followers at some point in time to be considered for associtation.

    It first labels the lanes, then builds the leader-follower dataframe, and filters out the vehicles that are not in the mainline lanes.

    Args:
        joined_df (pl.DataFrame): _description_
        mainline_net (RoadNetwork): _description_
        full_net (RoadNetwork): _description_

    Returns:
        pl.DataFrame: _description_
    """
    return (
        joined_df.pipe(
            label_lanes_tree,
            full_network=full_net,
            kalman_network=mainline_net,
            lane_width=full_net.LANE_WIDTH,
            s_col="s",
        )
        .lazy()
        .sort(by=["epoch_time"])
        .set_sorted(["epoch_time"])
        .lazy()
        .pipe(
            build_leader_follower_entire_history_df,
            # build_leader_follower_no_sort,
            s_col="s",
            use_lane_index=True,
            max_s_gap=0.5 * 35,  # max headway of 0.5 seconds at 35 m/s
        )
        # only include vehicles that are in the mainline lanes and have a velocity greater than 0.5 m/s
        # .filter((pl.col("s_velocity") > 0.5) & (pl.col("s_velocity_leader") > 0.5))
        .filter(pl.col("lane_index") <= 1)
        .collect(streaming=True)
    )


def build_assoc_liklihood_distance(
    lf_df,
) -> pl.DataFrame:
    return lf_df.pipe(calc_assoc_liklihood_distance, gpu=GPU, dims=4, permute=False)


@cache_wrapper("match_df")
def build_match_df(
    lf_df: pl.DataFrame,
    joined_df: pl.DataFrame,
    assoc_cutoff: float = None,
) -> pl.DataFrame:
    return (
        lf_df
        # is this really necessary? Just filters the match for a certain tome
        .pipe(
            calculate_match_indexes,
        )
        .pipe(
            pipe_gate_headway_calc,
            window=20,  # this is two seconds
            association_dist_cutoff=assoc_cutoff,
        )
        .pipe(
            build_match_pipeline,
            traj_time_df=joined_df.group_by("object_id")
            .agg(
                pl.col("epoch_time").max().alias("epoch_time_max"),
            )
            .lazy(),
            assoc_cutoff=assoc_cutoff,
            assoc_cutoff_pred=assoc_cutoff,
        )
        .collect(streaming=True)
    )


def _get_ordered_combinations(cc):
    return [
        (int(start), int(end), veh_i) if start < end else (end, start, veh_i)
        for veh_i, cc_list in enumerate(cc)
        for start, end in combinations(cc_list, 2)
    ]


@cache_wrapper("assoc_df")
def build_graph(
    joined_df: pl.DataFrame, match_df: pl.DataFrame, cutoff: float = None
) -> pl.DataFrame:
    if "Arr" in str(match_df["pair"].dtype):
        match_df = match_df.with_columns(pl.col("pair").arr.to_list())

    # create the naive graph
    cc, G, assoc_df = joined_df.pipe(
        create_vehicle_ids,
        match_df,
    )
    combs = _get_ordered_combinations(cc)

    # after the graph creation, find clusters that have more than 2 tracklets
    need2_filt_ids = (
        assoc_df.with_columns(
            pl.count().over(["epoch_time", "vehicle_id"]).alias("count")
        )
        .filter(pl.col("count").max().over("vehicle_id") > 2)["vehicle_id"]
        .unique()
    )

    # create an overlap dataframe
    begin_end_df = joined_df.group_by("object_id").agg(
        pl.col("epoch_time").min().alias("begin_time"),
        pl.col("epoch_time").max().alias("end_time"),
    )

    # create a permute dataframe
    permute_df = (
        pl.DataFrame(
            combs,
            schema={"start": pl.UInt64, "end": pl.UInt64, "vehicle_index": pl.Int64},
        )
        .filter(pl.col("vehicle_index").is_in(need2_filt_ids))
        .join(begin_end_df.rename({"object_id": "start"}), on="start", how="left")
        .join(
            begin_end_df.rename({"object_id": "end"}),
            on="end",
        )
        # filter for overlap in time
        .filter(
            pl.min_horizontal(pl.col("end_time"), pl.col("end_time_right"))
            > pl.max_horizontal(pl.col("begin_time"), pl.col("begin_time_right"))
        )
        .with_columns(
            # sort the object id and leader
            pl.when(pl.col("start") < pl.col("end"))
            .then(pl.concat_list([pl.col("start"), pl.col("end")]))
            .otherwise(pl.concat_list([pl.col("end"), pl.col("start")]))
            .alias("pair")
        )
        .with_columns(pl.col("pair").hash().alias("pair_hash"))
        .with_columns(
            pl.concat_str(
                [pl.col("pair").list.get(0), pl.col("pair").list.get(1)], separator="-"
            ).alias("pair_str")
        )
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("association_distance_filt"),
        )
        .join(
            match_df.with_columns(
                pl.concat_str(
                    [pl.col("pair").list.get(0), pl.col("pair").list.get(1)],
                    separator="-",
                ).alias("pair_str"),
                (pl.col("prediction") | pl.col("prediction_leader")).alias(
                    "prediction"
                ),
            ).select(["pair_str", "prediction"]),
            on="pair_str",
            how="left",
        )
        .with_columns(pl.col("prediction").fill_null(False))
    )

    need2compute = permute_df.filter(
        pl.col("association_distance_filt").is_null()
    ).select("start", "end", "pair_hash", "vehicle_index", "prediction")

    cols = [
        pl.col("object_id"),
        "epoch_time",
        "s",
        "lane",
        "d",
        "s_velocity",
        "d_velocity",
        "front_s",
        "back_s",
        "P",
        "prediction",
    ]

    need2compute = (
        need2compute.lazy()
        .join(
            joined_df.lazy().select(cols).rename({"object_id": "start"}),
            on="start",
        )
        .join(
            joined_df.lazy().select(cols).rename({"object_id": "end"}),
            on=["end", "epoch_time"],
            suffix="_leader",
        )
        .filter(
            (
                pl.sum_horizontal("prediction", "prediction_right", "prediction_leader")
                == 0
            )
            | (
                pl.col("prediction")
                & (pl.col("prediction_right") | pl.col("prediction_leader"))
            )
        )
        .collect()
        .pipe(calc_assoc_liklihood_distance, gpu=GPU, dims=4, permute=False, maha=False)
        .lazy()
        .with_columns(
            pl.col("association_distance")
            .rolling_mean(window_size=20, min_periods=1)
            .over("pair_hash")
        )
        .group_by("pair_hash")
        .agg(
            pl.col("association_distance").mean().alias("association_distance_filt"),
        )
        .collect()
    )

    permute_df = permute_df.update(need2compute, on="pair_hash", how="left").filter(
        pl.col("association_distance_filt").is_not_null()
    )

    # permute_df.write_parquet(
    #     "tmp/permute_df.parquet", use_pyarrow=True,
    # )

    big_G = nx.Graph()

    for d in permute_df.select(
        ["start", "end", "association_distance_filt"]
    ).to_dicts():
        big_G.add_edge(d["start"], d["end"], weight=d["association_distance_filt"])

    cropped_G = nx.subgraph_view(G, filter_node=lambda x: x in big_G.nodes)

    bad_edges = nx.difference(big_G, cropped_G).copy()
    # update the weights from bigG
    for u, v in bad_edges.edges:
        bad_edges[u][v]["weight"] = big_G[u][v]["weight"]

    remove_edges = []

    for veh in tqdm(permute_df["vehicle_index"].unique()):
        remove_edges.extend(
            walk_graph_removals(
                cropped_G.subgraph(cc[veh]).copy(),
                max_removals=20,
                cutoff=chi2.ppf(0.999, 4),
                big_G=big_G,
            )
        )

    mainG = G.copy()
    for edge in remove_edges:
        mainG.remove_edge(*edge)

    return make_graph_based_ids(assoc_df.drop("vehicle_id"), mainG)


@cache_wrapper("fusion_df")
def _build_fusion_df(assoc_df: pl.DataFrame, prediction_length: float) -> pl.DataFrame:
    return assoc_df.pipe(
        build_fusion_df, prediction_length=prediction_length, max_vehicle_num=3
    ).collect(streaming=True)


@cache_wrapper("info_df")
def _build_info_df(
    fusion_df: pl.DataFrame,
):
    outer_df = (
        fusion_df.select(
            [
                "front_s",
                "s",
                "back_s",
                "s_velocity",
                "d",
                "d_velocity",
                "P",
                "vehicle_id",
                "time_index",
                "vehicle_time_index_int",
                "prediction",
                "length_s",
            ]
        )
        .lazy()
        .filter(~pl.col("prediction"))
    )

    outer_df = (
        (
            outer_df.join(
                outer_df,
                on=["time_index", "vehicle_id"],
                how="outer",
                suffix="_leader",
            )
        )
        .with_columns(
            pl.struct(
                [
                    pl.min_horizontal(
                        [
                            pl.col("vehicle_time_index_int"),
                            pl.col("vehicle_time_index_int_leader"),
                        ]
                    ).alias("one"),
                    pl.max_horizontal(
                        [
                            pl.col("vehicle_time_index_int"),
                            pl.col("vehicle_time_index_int_leader"),
                        ]
                    ).alias("two"),
                ]
            ).alias("vehicle_time_struct"),
        )
        # .collect()
        .filter(
            (
                pl.col("vehicle_time_index_int")
                .cum_count()
                .over(["vehicle_id", "time_index", "vehicle_time_struct"])
                < 1
            )
            & (
                pl.col("vehicle_time_index_int")
                != pl.col("vehicle_time_index_int_leader")
            )
        )
        .collect(streaming=True)
        .pipe(
            calc_assoc_liklihood_distance,
            gpu=GPU,
            dims=4,
        )
        .group_by(["vehicle_id", "time_index"])
        .agg(pl.col("association_distance").max())
    )

    return outer_df


@cache_wrapper("fused_df")
def fuse_df(fusion_df: pl.DataFrame) -> pl.DataFrame:
    return batch_join(
        fusion_df,
        method="ImprovedFastCI",
        batch_size=10_000 if not GPU else 3_000,
        gpu=GPU,
        s_col="s",
    )


def smooth_df(
    fused_df: pl.DataFrame,
) -> pl.DataFrame:
    return rts_smooth(
        fused_df,
        batch_size=10_000 if not GPU else 10_000,
        gpu=GPU,
        s_col="s",
    )


def augment_merged_df(
    smoothed_df: pl.DataFrame,
    fusion_df: pl.DataFrame,
    outer_df: pl.DataFrame,
    full_net: RoadNetwork,
    mainline_net: RoadNetwork,
) -> None:
    prediction_tracker = (
        fusion_df.select(
            [
                "vehicle_id",
                "time_index",
                "prediction",
            ]
        )
        .lazy()
        .group_by(["vehicle_id", "time_index"])
        .agg(pl.col("prediction").all().alias("prediction"))
        .collect()
    )

    distance_adjust_df = (
        fusion_df.group_by(
            [
                "vehicle_id",
                "time_index",
            ]
        )
        .agg(
            pl.col("distanceToFront_s")
            .filter(pl.col("approaching"))
            .mean()
            .alias("distanceToFront_s"),
            pl.col("distanceToBack_s")
            .filter(~pl.col("approaching"))
            .mean()
            .alias("distanceToBack_s"),
            pl.col("distanceToFront_s").mean().alias("distanceToFront_s_all"),
            pl.col("distanceToBack_s").mean().alias("distanceToBack_s_all"),
        )
        .with_columns(
            pl.when(pl.col("distanceToFront_s").is_null())
            .then(pl.col("distanceToFront_s_all"))
            .otherwise(pl.col("distanceToFront_s"))
            .alias("distanceToFront_s"),
            pl.when(pl.col("distanceToBack_s").is_null())
            .then(pl.col("distanceToBack_s_all"))
            .otherwise(pl.col("distanceToBack_s"))
            .alias("distanceToBack_s"),
        )
        .drop(["distanceToFront_s_all", "distanceToBack_s_all"])
    )

    return (
        smoothed_df.drop("lane_index")
        .pipe(
            label_lanes_tree,
            full_network=full_net,
            kalman_network=mainline_net,
            lane_width=mainline_net.LANE_WIDTH,
            s_col="s_smooth",
            d_col="d_smooth",
        )
        .lazy()
        .join(
            fusion_df.lazy()
            .select(["time_index", "vehicle_id", "object_id", "length_s"])
            .group_by(["time_index", "vehicle_id"])
            .agg(pl.col("object_id"), pl.col("length_s").mean()),
            on=["time_index", "vehicle_id"],
            how="left",
        )
        .join(
            outer_df.lazy().select(
                ["time_index", "vehicle_id", "association_distance"]
            ),
            on=["time_index", "vehicle_id"],
            how="left",
        )
        .join(
            distance_adjust_df.lazy(),
            on=["vehicle_id", "time_index"],
            how="left",
        )
        .join(
            prediction_tracker.lazy(),
            on=["vehicle_id", "time_index"],
            how="left",
        )
        .with_columns(
            (pl.col("ci_s") + pl.col("distanceToFront_s")).alias("ci_front_s"),
            (pl.col("ci_s") + pl.col("distanceToBack_s")).alias("ci_back_s"),
            (pl.col("s_smooth") + pl.col("distanceToFront_s")).alias("front_s_smooth"),
            (pl.col("s_smooth") + pl.col("distanceToBack_s")).alias("back_s_smooth"),
        )
        .with_columns(
            ((pl.col("ci_front_s") + pl.col("ci_back_s")) / 2).alias("ci_s"),
            ((pl.col("front_s_smooth") + pl.col("back_s_smooth")) / 2).alias(
                "s_smooth"
            ),
        )
        .collect()
    )


def add_lane_info(
    merged_df: pl.DataFrame,
    mainline_net: RoadNetwork,
) -> pl.DataFrame:
    transformations = [
        ("front_s_smooth", "d_smooth", "front_x_smooth", "front_y_smooth"),
        ("back_s_smooth", "d_smooth", "back_x_smooth", "back_y_smooth"),
        ("s_smooth", "d_smooth", "centroid_x_smooth", "centroid_y_smooth"),
        ("ci_front_s", "ci_d", "ci_front_x", "ci_front_y"),
        ("ci_back_s", "ci_d", "ci_back_x", "ci_back_y"),
        ("ci_s", "ci_d", "ci_centroid_x", "ci_centroid_y"),
    ]

    for s_col, d_col, x_col, y_col in transformations:
        merged_df = (
            merged_df.drop([x_col, y_col])
            .pipe(
                mainline_net.frenet2xy,
                lane_col="lane",
                s_col=s_col,
                d_col=d_col,
            )
            .drop(["s", "angle"])
            .rename({"x_lane_point": x_col, "y_lane_point": y_col})
        )

    return merged_df


def build_small_df(merged_df: pl.DataFrame, full_net: RoadNetwork) -> pl.DataFrame:
    return (
        merged_df.drop(["x_lane", "y_lane", "lane"])
        .filter(pl.col("front_x_smooth").is_not_nan())
        .pipe(
            full_net.map_to_lane, utm_x_col="front_x_smooth", utm_y_col="front_y_smooth"
        )
        .rename({"name": "lane"})
        .filter(
            pl.col("lane").is_not_null() & pl.col("s_velocity_smooth").is_not_null()
        )
        .pipe(CalibratedRadar.add_cst_timezone)
        .select(
            [
                pl.col("epoch_time_cst").alias("epoch_time"),
                "s_smooth",
                "length_s",
                "s_velocity_smooth",
                "s_accel_smooth",
                "vehicle_id",
                "lane",
                "x_lane",
                "y_lane",
                pl.col("association_distance").mean().over("vehicle_id"),
                pl.col("object_id").list.len().max().over("vehicle_id"),
                (
                    (
                        pl.col("epoch_time") - pl.col("epoch_time").min()
                    ).dt.total_milliseconds()
                    / 1000
                )
                .over("vehicle_id")
                .alias("vehicle_time"),
            ]
        )
        .with_columns(pl.col(pl.FLOAT_DTYPES).cast(pl.Float32))
        .filter((pl.col("vehicle_time").max().over("vehicle_id") > 5))
        .sort("epoch_time")
    )


# def apply_lowpass_filter(
#     df: pl.DataFrame,
# ) -> pl.DataFrame:
#     return df.pipe(
#         butter_lowpass_filter_plot, "s_velocity_smooth", "vehicle_id"
#     ).with_columns(
#         (pl.col("s_velocity_smooth_lowpass").diff() / pl.col("vehicle_time").diff())
#         .backward_fill(1)
#         .over("vehicle_id")
#         .alias("lowpass_accel")
#     )


@click.command()
@click.argument("raw_data_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("calibration_yaml", type=click.Path())
@click.option(
    "--prediction_length",
    default=4,
    help="The prediction length in seconds",
)
@click.option(
    "--cache-dir",
    default=os.environ.get("CACHE_DIR"),
    help="The prediction length in seconds",
)
@click.option(
    "--lowpass-filter",
    default=False,
    help="Whether to apply a lowpass filter to the data",
    is_flag=True,
)
@click.option(
    "--no-cache",
    default=False,
    help="Turn caching off",
    is_flag=True,
)
@click.option(
    "--build-small",
    default=False,
    help="Build small dataframe",
    is_flag=True,
)
def run(
    raw_data_path,
    output_path,
    calibration_yaml,
    prediction_length,
    cache_dir,
    lowpass_filter,
    no_cache,
    build_small,
):
    if cache_dir is not None:
        os.environ["CACHE_DIR"] = cache_dir

    if lowpass_filter:
        raise NotImplementedError("Lowpass filter is not implemented")

    global CACHE
    CACHE = not no_cache

    mainline_net, full_net, radar_obj = create_helper_objs(calibration_yaml)

    radar_df = build_radar_df(
        open_radar_data(
            raw_data_path,
            radar_obj=radar_obj,
            mainline_net=mainline_net,
            build_small_df=build_small,
        ),
        prediction_length=prediction_length,
    )

    filtered_df = imm_filter(
        radar_df,
    )

    lf_df = build_lf_df(
        filtered_df,
        mainline_net,
        full_net,
    )

    match_df = build_match_df(
        build_assoc_liklihood_distance(lf_df),
        filtered_df,
        assoc_cutoff=chi2.ppf(0.95, 4),
    )

    fusion_df = _build_fusion_df(
        build_graph(
            filtered_df,
            match_df,
            cutoff=chi2.ppf(0.95, 4),
        ),
        prediction_length=4,
    )

    smoothed_df = smooth_df(fuse_df(fusion_df))

    final_df = augment_merged_df(
        smoothed_df,
        fusion_df,
        _build_info_df(fusion_df),
        full_net,
        mainline_net,
    )

    small_final_df = add_lane_info(final_df, mainline_net=mainline_net)
    # if lowpass_filter:
    #     small_final_df = apply_lowpass_filter(small_final_df)
    if build_small:
        # fit this inside GitHub's 100mb restriction
        import polars.selectors as cs

        small_final_df.drop(cs.contains("ci")).drop(
            cs.contains("index"), "vehicle_id_int"
        ).with_columns(pl.col(cs.FLOAT_DTYPES).cast(pl.Float32)).write_parquet(
            output_path, use_pyarrow=True
        )
    else:
        small_final_df.pipe(radar_obj.add_cst_timezone).write_parquet(
            output_path, use_pyarrow=True
        )


if __name__ == "__main__":
    run()
