from typing import Dict
import numpy as np
import polars as pl
from src.geometry import RoadNetwork
from src.pipelines.utils import lazify, timeit
import torch
import pomegranate as pmg


@lazify
@timeit
def label_lane(
    df: pl.DataFrame,
    right_lane_center: float,
    lane_width: float,
    d_col: str = "d",
) -> pl.DataFrame:
    half_lane_width = lane_width / 2
    left_lane_center_eb = right_lane_center + lane_width
    return df.with_columns(
        # pl.when(pl.col("lane").str.contains("W"))
        # .then(
        pl.when(
            pl.col("d").is_between(
                right_lane_center - half_lane_width,
                right_lane_center + half_lane_width,
            )
        )
        .then(0)
        .otherwise(
            pl.when(
                pl.col("d").is_between(
                    left_lane_center_eb - half_lane_width,
                    left_lane_center_eb + half_lane_width,
                    closed="right",
                )
            )
            .then(1)
            .otherwise(None)
        )
        .alias("lane_index")
        # .otherwise(
        #     pl.when(
        #         pl.col("d").is_between(
        #             right_lane_center - half_lane_width,
        #             right_lane_center + half_lane_width,
        #         )
        #     )
        #     .then(0)
        #     .otherwise(
        #         pl.when(
        #             pl.col("d").is_between(
        #                 left_lane_center_eb - half_lane_width,
        #                 left_lane_center_eb + half_lane_width,
        #             )
        #         )
        #         .then(1)
        #         .otherwise(None)
        #     )
        # )
        # .alias("lane_index")
    )


def label_lanes_tree(
    df: pl.DataFrame,
    full_network: RoadNetwork,
    kalman_network: RoadNetwork,
    lane_width: float,
    s_col: str = "s",
    d_col: str = 'd',
    time_col: str = "epoch_time",
) -> pl.DataFrame:

    return (
        df
        # .lazy()
        .sort(s_col)
        .join_asof(
            kalman_network.map_to_lane(
                full_network.df.filter(pl.col("name").str.ends_with("2")),
                directed=False,
                dist_upper_bound=10,
                utm_x_col="x",
                utm_y_col="y",
            )
            # .lazy()
            .with_columns(
                (pl.col("d") / 2).alias("d_cutoff"),
                pl.col("s_lane").cast(pl.Float32),
                pl.col("d").alias("d_other_center"),
            )
            .sort("s_lane")
            .select(["name_lane", "s_lane", "d_cutoff", "d_other_center"]),
            left_on=s_col,
            right_on="s_lane",
            strategy="nearest",
            tolerance=kalman_network.step_size * 2,
            by_left="lane",
            by_right="name_lane",
        )
        .with_columns(
            pl.col("d_cutoff").forward_fill().over("lane"),
            pl.col("d_other_center").forward_fill().over("lane"),
            pl.lit(None, dtype=pl.UInt16).alias("lane_index"),
        )
        # .drop("s_lane", "name_lane")
        .with_columns(
            pl.when(pl.col(d_col) < pl.col("d_cutoff"))
            .then(
                pl.when(pl.col(d_col) > (-1 * lane_width / 2))
                .then(pl.lit(0))
                .otherwise(pl.col("lane_index"))
            )
            .otherwise(
                pl.when(pl.col(d_col) < ((lane_width / 2) + pl.col("d_other_center")))
                .then(pl.lit(1))
                .otherwise(pl.col("lane_index"))
            )
            .alias("lane_index")
        )
        .sort(time_col)
        .set_sorted(time_col)
        # .collect(streaming=True)
    )
