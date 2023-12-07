from datetime import timedelta
import numpy as np
import polars as pl

from src.radar import BasicRadar, CalibratedRadar
from src.filters.vectorized_kalman import IMMFilter


def timeit(func):
    def timed(*args, **kwargs):
        import time

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        print(
            "function: {} took: {} seconds".format(func.__name__, end_time - start_time)
        )

        return result

    return timed


def lazify(func):
    # if the input is a lazy frame, then return a lazy frame
    # otherwise return a regular frame
    def lazy_func(*args, **kwargs):
        if isinstance(args[0], pl.LazyFrame):
            res = func(*args, **kwargs)
            if isinstance(res, pl.DataFrame):
                return res.lazy()
            return res
        else:
            return func(*args, **kwargs)

    return lazy_func


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
            # (pl.col("f32_velocityInDir_mps")).alias(
            #     "s_velocity"
            # ),
            # (pl.lit(0)).alias(
            #     "d_velocity"
            # ),
        )
        .with_columns(
            [
                (pl.col("f32_distanceToFront_m") * pl.col("angle_diff").cos()).alias(
                    "distanceToFront_s"
                ),
                (pl.col("f32_distanceToBack_m") * pl.col("angle_diff").cos()).alias(
                    "distanceToBack_s"
                ),
                # do the vehicle length
                (pl.col("f32_length_m") * pl.col("angle_diff").cos()).alias("length_s"),
            ]
        )
    )


@lazify
@timeit
def filter_short_trajectories(
    df: pl.DataFrame,
    minimum_distance_m: float = 10,
    minimum_duration_s: float = 2,
) -> pl.DataFrame:
    return (
        df.with_columns(
            (
                (
                    pl.col("epoch_time").max() - pl.col("epoch_time").min()
                ).dt.milliseconds()
                / 1e3
            )
            .over(["object_id", "lane"])
            .alias("duration_s"),
            (pl.col("s").max() - pl.col("s").min())
            .over(["object_id", "lane"])
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
    df: pl.DataFrame, seconds: float = 4, dt: float = 0.1
) -> pl.DataFrame:
    # take the last data point from the radar, and then extend it
    df = df.lazy()

    # get the datatype of the epoch_time column
    time_dt = df.dtypes[df.columns.index("epoch_time")]

    test = (
        df.lazy()
        .group_by("object_id")
        .agg(
            pl.col("epoch_time").min(),
            pl.col("epoch_time").max().alias("max_time"),
            (
                (
                    pl.col("epoch_time").max() - pl.col("epoch_time").min()
                ).dt.milliseconds()
                / 1e3
            ).alias("total_duration_ms"),
        )
        .with_columns(
            ((pl.col("total_duration_ms") + seconds) / dt).cast(int).alias("n_steps"),
            pl.lit(dt).alias("dt"),
        )
        .with_columns(
            pl.col("dt").repeat_by(pl.col("n_steps")).alias("dt"),
        )
        .explode("dt")
        .with_columns(
            (
                pl.col("dt").cumsum().over("object_id") * 1e3
                + pl.col("epoch_time").cast(float)
            )
            .cast(time_dt)
            .alias("epoch_time")
        )
        .with_columns((pl.col("epoch_time") > pl.col("max_time")).alias("prediction"))
        .drop(["dt", "n_steps", "total_duration_ms", "max_time"])
        .join(
            df.with_columns(pl.lit(False).alias("missing_data")),
            on=["object_id", "epoch_time"],
            how="left",
        )
        .sort(by=["object_id", "epoch_time"])
        .with_columns(
            pl.col("missing_data").fill_null(True),
            pl.col("prediction").fill_null(False),
        )
        .with_columns(pl.col(set(df.columns) - {"prediction"}).forward_fill())
        .collect()
        .rechunk()
        .set_sorted(["object_id", "epoch_time"])
        .lazy()
    )

    return test


@lazify
@timeit
def add_timedelta(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.lazy()
        .sort(by=["object_id", "epoch_time"])
        .with_columns(
            (pl.col("epoch_time").diff().dt.total_milliseconds() * 1e-3)
            .over("object_id")
            .fill_null(0)
            .alias("time_diff")
        )
    )


@lazify
@timeit
def build_kalman_id(
    df: pl.DataFrame,
    split_time_delta: float = 4,
) -> pl.DataFrame:
    return (
        df.with_columns(
            (pl.col("time_diff") > split_time_delta).alias("split"),
        )
        .with_columns(
            pl.col("split").cumsum().over(["object_id", "lane"]).alias("trajectory_id"),
        )
        .with_columns(
            pl.struct(
                (pl.col("object_id"), pl.col("trajectory_id"), pl.col("lane")),
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
                        pl.col("s"),
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
            ]
        )
        .with_columns(
            pl.col("time_ind").max().over("kalman_id").alias("max_time"),
        )
        .filter(pl.col("max_time") > minimum_data_points)
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
