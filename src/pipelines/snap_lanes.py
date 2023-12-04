from typing import List
import polars as pl


from src.frenet import SplineLane
from src.radar import BasicRadar


def pipe_lanes(
    df: pl.DataFrame,
    radar_obj: BasicRadar,
    spline_lanes: List[SplineLane],
    distance_threshold: float = 1.6,
) -> pl.DataFrame:
    radar_obj = BasicRadar if radar_obj is None else radar_obj
    return (
        df.pipe(
            radar_obj.map_frenet_lanes,
            lanes=spline_lanes,
            distance_threshold=distance_threshold,
            drop_outside=False,
        )
        .pipe(radar_obj.atan2, x_col="ds_du_x", y_col="ds_du_y", out_col="s_angle")
        .pipe(radar_obj.rad_to_degrees, rad_col="s_angle", out_col="s_angle_deg")
        .with_columns(
            [
                # calulate the difference in angle between the heading and the lane angle
                (
                    180
                    - abs(
                        abs(pl.col("s_angle_deg") - pl.col("direction_degrees")) - 180
                    )
                ).alias("angle_diff"),
            ]
        )
    )


def pipe_lanebounce_fix(
    df: pl.DataFrame,
    spline_lanes: List[SplineLane],
    radar_obj: BasicRadar = None,
    angle_threshold: float = 30,
    min_duration: float = 0.5,
    lane_col_out: str = "lane",
) -> pl.DataFrame:
    radar_obj = BasicRadar if radar_obj is None else radar_obj

    local_df = (
        # add lane group, which is a unique identifier for each continuouse lane - vehicle pair
        df.pipe(radar_obj.add_lane_group)
        .lazy()
        .with_columns(
            [
                (
                    (pl.col("epoch_time").max() - pl.col("epoch_time").min()).cast(
                        pl.Float32()
                    )
                    / 1000
                )
                .over(["object_id", "lane", "lane_group"])
                .alias("duration"),
            ]
        )
        .with_columns(
            [
                pl.when(
                    (pl.col("angle_diff") < angle_threshold)
                    & (pl.col("duration") > min_duration),
                )
                .then(pl.col(col))
                .otherwise(
                    pl.lit(None, dtype=pl.Utf8() if col == "lane" else pl.Float32())
                )
                .alias(col)
                for col in ["lane", "s", "min_d", "ds_du_x", "ds_du_y", "angle_diff"]
            ]
        )
        .sort(["object_id", "epoch_time"])
        .with_columns(
            [
                pl.col("lane").forward_fill().over("object_id").alias("lane_forward"),
                pl.col("lane").backward_fill().over("object_id").alias("lane_backward"),
            ]
        )
        .with_columns(
            [
                pl.when(
                    (
                        (pl.col("lane_forward") == pl.col("lane_backward"))
                        & (pl.col("duration") < min_duration)
                    ),
                )
                .then(pl.col("lane_forward"))
                .otherwise(pl.col("lane"))
                .alias("lane_corrected"),
            ]
        )
        # .pipe(BasicRadar.add_lane_group, lane_col="lane_corrected")
        .collect()
        .pipe(
            radar_obj.map_frenet_lane_constrained,
            lanes=spline_lanes,
            filter_expr=(
                ((pl.col("s") == 0) | pl.col("s").is_null())
                & pl.col("lane_corrected").is_not_null()
            ),
            lane_col="lane_corrected",
            distance_threshold=10,
        )
    )

    #
    if lane_col_out in local_df.columns:
        local_df = local_df.drop(lane_col_out)

        local_df = local_df.rename(
            {
                "lane_corrected": lane_col_out,
            }
        )
    return local_df
