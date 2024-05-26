import polars as pl
import utm

from src.geometry import RoadNetwork
from src.pipelines.lane_classification import label_lanes_tree

GLE_m = 4.93522
DIST_2_FRONT_m = 2.91
DIST_2_BACK_m = 1.99


def open_probe_df(path: str) -> pl.DataFrame:
    df = (
        pl.scan_parquet(path)
        .sort(
            "gps_time",
        )
        .with_row_count(name="seq")
        .collect()
    )

    x, y, _, _ = utm.from_latlon(
        latitude=df["lat"].to_numpy(),
        longitude=df["lon"].to_numpy(),
    )

    return (
        df.with_columns(
            x=x,
            y=y,
            gps_time=pl.col("gps_time").dt.truncate("100ms"),
        )
        .with_columns(
            pl.col("x").diff().alias("dx"),
            pl.col("y").diff().alias("dy"),
            pl.lit(GLE_m).alias("length_s"),
        )
        .with_columns(
            (pl.col("dx").pow(2) + pl.col("dy").pow(2)).sqrt().alias("dist"),
            # find the heading
            pl.arctan2(pl.col("dy"), pl.col("dx")).alias("heading"),
        )
        .with_columns(
            pl.col("dist").cum_sum().alias("cum_dist"),
            # make the front x/y position
            (pl.col("x") + (DIST_2_FRONT_m * pl.col("heading").cos())).alias("front_x"),
            (pl.col("y") + (DIST_2_FRONT_m * pl.col("heading").sin())).alias("front_y"),
            (pl.col("x") - (DIST_2_BACK_m * pl.col("heading").cos())).alias("back_x"),
            (pl.col("y") - (DIST_2_BACK_m * pl.col("heading").sin())).alias("back_y"),
        )
        .with_columns(
            #
            (
                (pl.col("cum_dist").shift(-1) - pl.col("cum_dist").shift(1)).abs()
                / (
                    (
                        pl.col("gps_time").shift(-1) - pl.col("gps_time").shift(1)
                    ).dt.total_milliseconds()
                    / 1e3
                )
            ).alias("speed"),
        )
    )


def snap_to_lane(
    df: pl.DataFrame,
    mainline_net: RoadNetwork,
    full_net: RoadNetwork,
) -> pl.DataFrame:
    return (
        df.pipe(
            mainline_net.map_to_lane,
            dist_upper_bound=mainline_net.LANE_WIDTH * mainline_net.LANE_NUM
            - (mainline_net.LANE_WIDTH / 2)
            + 0.5,  # centered on one of the lanes,
            utm_x_col="x",
            utm_y_col="y",
        )
        .filter(
            pl.col("name").is_not_null()
            & (pl.col("s") > 150)
            & (pl.col("s") < (pl.col("s").max() - 150)).over("name")
        )
        .rename(
            {
                "name": "lane",
                "angle": "heading_lane",
            }
        )
        .with_columns((pl.col("heading") - pl.col("heading_lane")).alias("angle_diff"))
        .with_columns(
            # find the portion of velocity that is in the direction of the lane
            pl.arctan2(pl.col("angle_diff").sin(), pl.col("angle_diff").cos()).alias(
                "angle_diff"
            )
        )
        .with_columns(
            (pl.col("speed") * pl.col("angle_diff").cos()).alias("s_velocity"),
            (pl.col("speed") * pl.col("angle_diff").sin()).alias("d_velocity"),
            ((DIST_2_FRONT_m * pl.col("angle_diff").cos()) + pl.col("s")).alias(
                "front_s"
            ),
            (pl.col("s") - (DIST_2_BACK_m * pl.col("angle_diff").cos())).alias(
                "back_s"
            ),
            # do the vehicle length
            (GLE_m * pl.col("angle_diff").cos()).alias("length_s"),
            pl.col("s").cast(pl.Float32).alias("s"),
        )
        .pipe(
            label_lanes_tree,
            full_network=full_net,
            kalman_network=mainline_net,
            lane_width=mainline_net.LANE_WIDTH,
            s_col="s",
            time_col="gps_time",
        )
    )


def build_continous_segment_df(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("lane").fill_null(""),
        )
        .with_columns(
            (
                (pl.col("gps_time").diff().dt.total_seconds() > 10)
                | (pl.col("lane").shift(1) != pl.col("lane"))
            ).alias("sequence"),
        )
        .with_columns(
            (pl.col("sequence").cum_sum() * (pl.col("lane") != "")).alias(
                "sequence_id"
            ),
        )
        .filter(pl.col("sequence_id") != 0)
    )


def crop_quantile_df(df: pl.DataFrame, crop_df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.join(
            crop_df.with_columns(pl.col("lane_index").cast(pl.UInt16)),
            on=["lane", "lane_index"],
            how="left",
        )
        .filter(
            pl.col("front_s").is_between(pl.col("quanitile_5"), pl.col("quanitile_95"))
        )
        .drop(["quanitile_5", "quanitile_95"])
        .sort("gps_time")
        .with_columns(
            pl.col("front_s")
            .diff()
            .sum()
            .over(["sequence_id"])
            .alias("sequence_distance"),
        )
    )
