from typing import Dict
import numpy as np
import polars as pl
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
                    closed='right'
                )
            )
            .then(1)
            .otherwise(None)
        ).alias("lane_index")
        
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
