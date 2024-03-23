import numpy as np
import polars as pl
from src.pipelines.utils import lazify, timeit
from typing import Tuple


@lazify
@timeit
def prepare_frenet_measurement(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    The Frenet measurement vector is: 
        \begin{align}
            \begin{bmatrix}
                s \\
                \dot{s} \\
                d \\
                \dot{d} \\
            \end{bmatrix}
    
    Args:
        df (pl.DataFrame): the input dataframe

    Returns:
        pl.DataFrame: the dataframe with the frenet measurement vector
    """

    # make sure that all the require columns are there
    required_cols = {
        "s",
        "lane",
        "heading_utm",
        "heading_lane",
        "f32_velocityInDir_mps",
        "f32_distanceToFront_m",
        "f32_distanceToBack_m",
        "f32_length_m",
    }

    assert required_cols.issubset(
        df.columns
    ), f"Missing columns: {required_cols - set(df.columns)}"

    return (
        df.lazy()
        .with_columns(
            (pl.col("heading_utm") - pl.col("heading_lane")).alias("angle_diff")
        )
        .with_columns(
            # find the portion of velocity that is in the direction of the lane
            pl.arctan2(pl.col("angle_diff").sin(), pl.col("angle_diff").cos()).alias(
                "angle_diff"
            )
        )
        .with_columns(
            pl.when(
                pl.col("angle_diff").abs() > np.deg2rad(30),
            )
            .then(pl.lit(None))
            .otherwise(pl.col("angle_diff"))
            .interpolate()
            .over(["object_id", "lane"])
            .alias("angle_diff")
        )
        .filter(pl.col("angle_diff").is_not_null())
        .with_columns(
            (pl.col("f32_velocityInDir_mps") * pl.col("angle_diff").cos()).alias(
                "s_velocity"
            ),
            (pl.col("f32_velocityInDir_mps") * pl.col("angle_diff").sin()).alias(
                "d_velocity"
            ),
            (pl.col("f32_distanceToFront_m") * pl.col("angle_diff").cos()).alias(
                "distanceToFront_s"
            ),
            (pl.col("f32_distanceToBack_m") * pl.col("angle_diff").cos()).alias(
                "distanceToBack_s"
            ),
            # do the vehicle length
            (pl.col("f32_length_m") * pl.col("angle_diff").cos()).alias("length_s"),
        )
    )


@lazify
@timeit
def filter_short_trajectories(
    df: pl.DataFrame,
    minimum_distance_m: float = 10,
    minimum_duration_s: float = 2,
    vehicle_id_col: str = "kalman_id",
) -> pl.DataFrame:
    return (
        df.with_columns(
            (
                (
                    pl.col("epoch_time").max() - pl.col("epoch_time").min()
                ).dt.total_milliseconds()
                / 1e3
            )
            .over(vehicle_id_col)
            .alias("duration_s"),
            (pl.col("s").max() - pl.col("s").min())
            .over(vehicle_id_col)
            .alias("distance_m"),
        )
        .filter(
            (pl.col("duration_s") > minimum_duration_s)
            & (pl.col("distance_m") > minimum_distance_m)
        )
        .drop(["duration_s", "distance_m"])
    )


@lazify
@timeit
def build_extension(
    df: pl.DataFrame,
    seconds: float = 4,
    dt: float = 0.1,
    id_cols: Tuple[str] = ("kalman_id",),
) -> pl.DataFrame:
    # take the last data point from the radar, and then extend it
    # df = df.lazy()
    dt = dt * 1000
    return (
        df
        .lazy()
        .group_by(id_cols)
        .agg(
            pl.col("epoch_time").min(),
            pl.col("epoch_time").max().alias("max_time"),
            (
                (
                    pl.col("epoch_time").max() - pl.col("epoch_time").min()
                ).dt.milliseconds()
                / 1e3
            ).alias("total_duration_s"),
        )
        .with_columns(
            ((pl.col("total_duration_s") + seconds) / (dt / 1000)).cast(int).alias("n_steps"),
            # pl.lit(dt * 1000).cast(int).alias("dt"),
            pl.lit(dt).cast(int).alias("dt"),
        )
        .with_columns(
            pl.col("dt").repeat_by(pl.col("n_steps")).alias("dt"),
        )
        .explode("dt")
        .with_columns((pl.col("dt").cumsum()).cast(int).over(id_cols))
        .with_columns(
            (pl.duration(milliseconds="dt") + pl.col("epoch_time"))
            .dt.round(every="100ms")
            .alias("epoch_time")
        )
        .with_columns((pl.col("epoch_time") > pl.col("max_time")).alias("prediction"))
        .drop(["dt", "n_steps", "total_duration_s", ])
        .join(
            df.lazy().with_columns(
                pl.lit(False).alias("missing_data"),
                pl.col("epoch_time").dt.round(every="100ms"),
            ),
            on=[*id_cols, "epoch_time"],
            how="left",
        )
        .filter(
            ~(
                (pl.col("object_id").cum_count() == 0) & (pl.col("object_id").is_null())
            ).over(id_cols)
        )
        .sort(by=[*id_cols, "epoch_time"])
        .with_columns(
            pl.col("missing_data").fill_null(True),
            pl.col("prediction").fill_null(False),
        )
        .with_columns(
            pl.col(
                set(df.columns) - {"prediction", "missing_data"}
            ).forward_fill()
        )
        .collect(streaming=True)
        .rechunk()
        .set_sorted(["object_id", "lane", "epoch_time"])
        .lazy()
    )


#     return test
# @lazify
# @timeit
# def build_extension(
#     df: pl.DataFrame, seconds: float = 4, dt: float = 0.1
# ) -> pl.DataFrame:
#     # take the last data point from the radar, and then extend it
#     df = df.lazy()

#     # get the datatype of the epoch_time column
#     time_dt = df.dtypes[df.columns.index("epoch_time")]

#     test = (
#         df.lazy()
#         .group_by(["object_id", "lane"])
#         .agg(
#             pl.col("epoch_time").min(),
#             pl.col("epoch_time").max().alias("max_time"),
#             (
#                 (
#                     pl.col("epoch_time").max() - pl.col("epoch_time").min()
#                 ).dt.milliseconds()
#                 / 1e3
#             ).alias("total_duration_ms"),
#         )
#         .with_columns(
#             ((pl.col("total_duration_ms") + seconds) / dt).cast(int).alias("n_steps"),
#             pl.lit(dt).alias("dt"),
#         )
#         .with_columns(
#             pl.col("dt").repeat_by(pl.col("n_steps")).alias("dt"),
#         )
#         .explode("dt")
#         .with_columns(
#             (
#                 pl.col("dt").cumsum().over(["object_id", "lane"]) * 1e3
#                 + pl.col("epoch_time").cast(float)
#             )
#             .cast(time_dt)
#             .alias("epoch_time")
#         )
#         .with_columns((pl.col("epoch_time") > pl.col("max_time")).alias("prediction"))
#         .drop(["dt", "n_steps", "total_duration_ms", "max_time"])
#         .join(
#             df.with_columns(pl.lit(False).alias("missing_data")),
#             on=["object_id", "epoch_time", "lane"],
#             how="left",
#         )
#         .sort(by=["object_id", "lane", "epoch_time"])
#         .with_columns(
#             pl.col("missing_data").fill_null(True),
#             pl.col("prediction").fill_null(False),
#         )
#         .with_columns(
#             pl.col(set(df.columns) - {"prediction", "missing_data"}).forward_fill()
#         )
#         .collect()
#         .rechunk()
#         .set_sorted(["object_id", "lane", "epoch_time"])
#         .lazy()
#     )

#     return test


@lazify
@timeit
def add_timedelta(df: pl.DataFrame, vehicle_id_col: str = "kalman_id") -> pl.DataFrame:
    return (
        df.lazy()
        .sort(by=["epoch_time"])
        .with_columns(
            (pl.col("epoch_time").diff().dt.total_milliseconds() * 1e-3)
            .backward_fill(1)
            .over(vehicle_id_col)
            .alias("time_diff")
        )
    )


@lazify
@timeit
def build_kalman_id(
    df: pl.DataFrame,
    split_time_delta: float = 4,
    max_seconds: float = 100,
) -> pl.DataFrame:
    return (
        df.with_columns(
            (pl.col("time_diff") > split_time_delta).alias("split"),
        )
        .with_columns(
            pl.col("split").cumsum().over(["object_id", "lane"]).alias("trajectory_id"),
            (
                (pl.col("epoch_time") - pl.col("epoch_time").min()).dt.total_seconds()
                // max_seconds
            )
            .over(["object_id", "lane"])
            .alias("epoch_time_group"),
        )
        .with_columns(
            pl.struct(
                (
                    pl.col("object_id"),
                    pl.col("trajectory_id"),
                    pl.col("lane"),
                    pl.col("epoch_time_group"),
                ),
                #    (pl.col("object_id"), pl.col("trajectory_id"), pl.col("lane"), )
            )
            .hash()
            .alias("kalman_id"),
        )
    )


@lazify
@timeit
def build_kalman_df(
    df: pl.DataFrame,
    minimum_data_points: int = 10,
    s_col: str = "s",
    derive_s_vel: bool = False,
) -> pl.DataFrame:
    kalman_df = (
        df.lazy()
        .select(
            [
                pl.col("kalman_id"),
                pl.col("epoch_time"),
                pl.col("time_diff"),
                pl.concat_list(
                    [
                        pl.col(s_col),
                        pl.col("s_velocity"),
                        pl.col("d"),
                        pl.col("d_velocity"),
                    ]
                )
                .cast(pl.Array(width=4, inner=pl.Float32))
                .alias("measurement"),
                pl.col("prediction"),
                pl.col("missing_data"),
                pl.col("epoch_time").cum_count().over("kalman_id").alias("time_ind"),
                pl.col("s_velocity"),
            ]
        )
        .with_columns(
            pl.col("time_ind").max().over("kalman_id").alias("max_time"),
        )
        .filter(pl.col("max_time") > minimum_data_points)
    )

    if derive_s_vel:
        if s_col != 's':
            print(f'Warning: s_dot might be noisy using {s_col}')
        df = df.with_columns(
            (pl.col(s_col).diff() / pl.col('time_diff')).reverse().rolling_mean(window_size=20).reverse().over('kalman_id').alias('s_velocity')
        )

    vehicle_inds = (
        kalman_df.group_by("kalman_id")
        .agg(pl.col("max_time").first())
        .sort("max_time")
        .with_row_count("vehicle_ind")
        .drop("max_time")
    )

    kalman_df = kalman_df.join(vehicle_inds, on="kalman_id", how="inner").with_columns(
        pl.col("vehicle_ind").alias("vehicle_ind")
    )

    return kalman_df.sort(by=["vehicle_ind", "time_ind"]).with_columns(
        pl.col("vehicle_ind").cast(int)
    )


@lazify
@timeit
def join_results(
    df: pl.DataFrame,
    kalman_df: pl.DataFrame,
    radar_df: pl.DataFrame,
) -> pl.DataFrame:
    radar_df_keep_cols = list(
        set(radar_df.columns) - set(df.columns) - set(kalman_df.columns)
    )
    radar_df_keep_cols.sort()

    return (
        df.lazy()
        .with_columns(pl.col(["time_ind", "vehicle_ind"]).cast(int))
        .join(
            kalman_df.lazy().with_columns(
                pl.col(["time_ind", "vehicle_ind"]).cast(int)
            ),
            on=["time_ind", "vehicle_ind"],
            how="inner",
        )
        .drop(["time_ind", "vehicle_ind"])
        .join(
            radar_df.lazy().select(radar_df_keep_cols + ["epoch_time", "kalman_id"]),
            on=["kalman_id", "epoch_time"],
            how="left",
        )
    )
