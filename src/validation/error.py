from datetime import timedelta
import polars as pl
from scipy.stats import pearsonr
import polars.selectors as cs


def offset_error(veh_df, radar_df, offset):
    # this function treis to eagerly join the two dataframes
    # and calculates the error between the two
    return (
        veh_df
        # .lazy()
        .with_columns(
            pl.col("epoch_time") + timedelta(seconds=offset),
        )
        .filter(pl.col("epoch_time").is_in(radar_df["epoch_time"].unique()))
        # .sort("s")
        .join(
            (radar_df),
            on=[
                "epoch_time",
                "lane",
            ],
            suffix="_radar",
        )
        .with_columns(
            # calculate the rms error
            ((pl.col("s") - pl.col("s_radar")) ** 2).alias("s_squared_error"),
            ((pl.col("speed") - pl.col("speed_radar")) ** 2).alias(
                "s_velocity_squared_error"
            ),
        )
        .group_by(["vehicle_id", "sequence_id"])
        .agg(
            (pl.col("s_squared_error").sum() / pl.count()).sqrt().alias("s_rmse"),
            (pl.col("s_velocity_squared_error").sum() / pl.count())
            .sqrt()
            .alias("s_velocity_rmse"),
        )
        .group_by("sequence_id")
        .agg(
            pl.col("vehicle_id")
            .gather(pl.col("s_rmse").arg_min())
            .alias("min_rmse_vehicle_id"),
            pl.col("vehicle_id")
            .gather(pl.col("s_velocity_rmse").arg_min())
            .alias("min_s_velocity_rmse_vehicle_id"),
            pl.col("s_rmse").min().alias("min_rmse"),
            pl.col("s_velocity_rmse").min().alias("min_s_velocity_rmse"),
        )
        .explode(["min_rmse_vehicle_id", "min_s_velocity_rmse_vehicle_id"])
    )


def find_opt_time(
    veh_df: pl.DataFrame,
    radar_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Find the optimal time offset between the two dataframes

    Args:
        veh_df (pl.DataFrame): The vehicle dataframe
        radar_df (pl.DataFrame): The radar dataframe

    Returns:
        pl.DataFrame: The optimal time offset
    """

    e_df = (
        veh_df.with_columns(
            pl.col("gps_time").cast(radar_df["epoch_time"].dtype).alias("epoch_time"),
            pl.col("front_s").cast(float).alias("s"),
            pl.col("lane_index").alias("lane_index"),
        )
        .sort("s")
        .set_sorted("s")
    )

    ee_df = (
        radar_df.with_columns(
            pl.col("front_s_smooth").cast(float).alias("s"),
            pl.col("s_velocity_smooth").alias("speed_radar"),
            # pl.col("lane_index").cast(pl.UInt32).alias("lane_index"),
        )
        .sort("s")
        .set_sorted("s")
    )

    error_summary = []

    for second_offset in range(-200, -100, 1):
        error_df = offset_error(
            e_df,
            ee_df,
            second_offset / 10,
        )

        error_summary.append(
            error_df.with_columns(
                pl.lit(second_offset / 10).alias("second_offset"),
            )
        )

    return pl.concat(error_summary)


def find_per_track_error(
    veh_df: pl.DataFrame,
    radar_df: pl.DataFrame,
    second_offset: float,
) -> pl.DataFrame:
    """
    Match the radar data to the vehicle data

    Args:
        veh_df (pl.DataFrame): The vehicle dataframe
        radar_df (pl.DataFrame): The radar dataframe
        second_offset (float): The time offset between the two dataframes

    Returns:
        pl.DataFrame: The matched data
    """
    return (
        veh_df.with_columns(
            (pl.col("gps_time") + timedelta(seconds=second_offset))
            .cast(radar_df["epoch_time"].dtype)
            .alias("epoch_time"),
            pl.col("front_s").cast(float).alias("front_s"),
            pl.col("back_s").cast(float).alias("back_s"),
            pl.col("s").cast(float).alias("s"),
        )
        .filter(pl.col("epoch_time").is_in(radar_df["epoch_time"].unique()))
        .join(
            radar_df.select(
                [
                    pl.col("front_s_smooth").cast(float).alias("front_s_radar"),
                    pl.col("back_s_smooth").cast(float).alias("back_s_radar"),
                    pl.col("s_smooth").cast(float).alias("s_radar"),
                    pl.col("s_velocity_smooth").alias("speed_radar"),
                    # pl.col("lane_index").cast(pl.UInt32).alias("lane_index"),
                    "lane_index",
                    "epoch_time",
                    "vehicle_id",
                    "lane",
                ]
            ),
            on=["epoch_time", "lane", "lane_index"],
            # how="outer",
        )
        .filter(pl.col("s_radar").is_not_nan())
        .with_columns(
            # calculate the rms error
            pl.min_horizontal(
                ((pl.col(col) - pl.col(f"{col}_radar")) ** 2)
                for col in ["front_s", "back_s", "s"]
            ).alias("s_squared_error"),
            ((pl.col("speed") - pl.col("speed_radar")) ** 2).alias(
                "s_velocity_squared_error"
            ),
        )
        .group_by(["vehicle_id", "sequence_id"])
        .agg(
            (pl.col("s_squared_error").sum() / pl.count()).sqrt().alias("s_rmse"),
            (pl.col("s_velocity_squared_error").sum() / pl.count())
            .sqrt()
            .alias("s_velocity_rmse"),
        )
    )


def paired_df(
    veh_df: pl.DataFrame,
    radar_df: pl.DataFrame,
    raw_radar_df: pl.DataFrame,
    imm_filter_df: pl.DataFrame,
    good_matches: pl.DataFrame,
    second_offset: float,
) -> pl.DataFrame:
    veh_df = veh_df.with_columns(
        (
            pl.col("gps_time").cast(radar_df["epoch_time"].dtype)
            + timedelta(seconds=second_offset)
        ).alias("epoch_time"),
        pl.col("lane_index").cast(int),
        pl.col("x").alias("centroid_x"),
        pl.col("y").alias("centroid_y"),
    )

    small_radar_df = radar_df.select(
        [
            "epoch_time",
            pl.col("^.*_smooth$"),
            pl.col("^ci_.*$"),
            pl.col("length_s").alias("length_s_smooth"),
            pl.col("length_s").alias("ci_length_s"),
            ((pl.col("s_velocity_smooth") ** 2) + (pl.col("d_velocity_smooth") ** 2))
            .sqrt()
            .alias("speed_smooth"),
            (
                (pl.col("ci_s_velocity") ** 2 + pl.col("ci_d_velocity") ** 2).sqrt()
            ).alias("ci_speed"),
            # other
            "vehicle_id",
            "lane",
            pl.col("lane_index").cast(int),
            "prediction",
        ]
    ).filter(pl.col("vehicle_id").is_in(good_matches["vehicle_id"].unique()))

    small_radar_df = small_radar_df.drop(cs.contains("_P"))

    small_raw_df = raw_radar_df.select(
        [
            "epoch_time",
            "front_s",
            "back_s",
            pl.col("centroid_s").alias("s"),
            "s_velocity",
            "d",
            "d_velocity",
            pl.col("f32_velocityInDir_mps").alias("speed_raw"),
            pl.col("utm_x").alias("centroid_x_raw"),
            pl.col("utm_y").alias("centroid_y_raw"),
            "vehicle_id",
            "object_id",
            "ip",
            "front_x",
            "front_y",
            "back_x",
            "back_y",
            pl.col("length_s").alias("length_s_raw"),
            "dist",
            "approaching",
        ]
    ).filter(pl.col("vehicle_id").is_in(good_matches["vehicle_id"].unique()))

    small_imm_df = imm_filter_df.select(
        [
            "epoch_time",
            pl.col("centroid_s").alias("s"),
            "front_s",
            "back_s",
            "s_velocity",
            "d",
            "d_velocity",
            "object_id",
            "vehicle_id",
            "front_x",
            "front_y",
            "back_x",
            "back_y",
            pl.col("centroid_x").alias("centroid_x_imm"),
            pl.col("centroid_y").alias("centroid_y_imm"),
            ((pl.col("s_velocity") ** 2) + (pl.col("d_velocity") ** 2))
            .sqrt()
            .alias("speed_imm"),
            (pl.col("front_s") - pl.col("back_s")).alias("length_s_imm"),
        ]
    )

    return (
        veh_df.join(
            small_radar_df,
            on=["epoch_time", "lane", "lane_index"],
            how="left",
            suffix="_rts",
        )
        .join(
            small_raw_df,
            on=["epoch_time", "vehicle_id"],
            how="left",
            suffix="_raw",
        )
        .join(
            small_imm_df,
            on=["epoch_time", "vehicle_id"],
            how="left",
            suffix="_imm",
        )
        .sort("epoch_time")
        .with_columns(
            pl.col("front_s")
            .diff()
            .sum()
            .over(["sequence_id", "vehicle_id"])
            .alias("merged_trajectory_length"),
            pl.col("front_s")
            .diff()
            .sum()
            .over(["sequence_id", "vehicle_id", "object_id"])
            .alias("raw_trajectory_length"),
            pl.col("front_s")
            .filter(pl.col("object_id_imm").is_not_null())
            .diff()
            .sum()
            .over(["sequence_id", "vehicle_id", "object_id_imm"])
            .alias("imm_trajectory_length"),
        )
    )


def element_error(
    df: pl.DataFrame,
    true_col: str,
    pred_col: str,
    out_col: str = None,
    pearsonr_col: str = None,
    mae_col: str = None,
) -> pl.DataFrame:
    out_col = out_col or f"{true_col}_se"
    pearsonr_col = pearsonr_col or f"{true_col}_pearsonr"
    mae_col = mae_col or f"{true_col}_mae"
    err_col = f"{true_col}_err"

    # this some f sh!t
    def lit_pearson_r(_df):
        return _df.with_columns(
            pl.lit(
                pearsonr(
                    _df[true_col].to_numpy(),
                    _df[pred_col].to_numpy(),
                )[0]
            ).alias(pearsonr_col)
        )

    return df.with_columns(
        ((pl.col(true_col) - pl.col(pred_col)) ** 2).alias(out_col),
        ((pl.col(true_col) - pl.col(pred_col)).abs()).alias(mae_col),
        (pl.col(true_col) - pl.col(pred_col)).alias(err_col),
    ).pipe(lit_pearson_r)


def calculate_errors(df, pred_col_func, additional_group_cols):
    # Filter to keep the first occurrence based on the cum_count condition
    filtered_df = df.filter(
        pl.col("vehicle_id")
        .cum_count()
        .over(["epoch_time", "vehicle_id"] + additional_group_cols)
        < 1
    )

    # Apply element_error for each pair of columns
    for col_suffix in [
        "front_s",
        "s_velocity",
        "back_s",
        "speed",
        "s",
        "length_s",
        "d",
    ]:
        true_col = f"{col_suffix}"
        pred_col = pred_col_func(col_suffix)
        filtered_df = filtered_df.pipe(
            element_error, true_col=true_col, pred_col=pred_col
        )

    # Calculate squared error for specific location columns
    for loc in [
        "front",
        "back",
        "centroid",
    ]:
        filtered_df = filtered_df.with_columns(
            (
                (pl.col(f"{loc}_x") - pl.col(pred_col_func(f"{loc}_x"))).pow(2)
                + (pl.col(f"{loc}_y") - pl.col(pred_col_func(f"{loc}_y"))).pow(2)
            ).alias(f"{loc}_xy_se")
        )

    return filtered_df


def group_and_aggregate(
    df,
    error_suffix,
    trajectory_length_col,
    vehicle_id_col,
    groupby_cols=["lane", "lane_index"],
):
    # Group by 'lane' and 'lane_index', then aggregate
    grouped_df = df.group_by(groupby_cols).agg(
        pl.col("^.*_se$").mean().name.map(lambda x: f"{x.replace('_se', '')}_mse"),
        pl.col("^.*_se$")
        .mean()
        .sqrt()
        .name.map(lambda x: f"{x.replace('_se', '')}_rmse"),
        pl.col("^.*_ae$").mean().name.map(lambda x: f"{x.replace('_ae', '')}_mae"),
        pl.col("^.*_pearsonr$").first(),
        (pl.col(trajectory_length_col) / pl.col("sequence_distance"))
        .mean()
        .alias("average_coverage_percent"),
        pl.col("sequence_id").n_unique().alias("n_sequences"),
        pl.col(vehicle_id_col).n_unique().alias("n_vehicles"),
    )

    return grouped_df


def build_error_df(
    error_df,
    groupby_col: str='method'
):
    # Calculate errors
    smoothed_pred_col_func = lambda col_suffix: f"{col_suffix}_smooth"  # noqa: E731
    smoothed_error_df = calculate_errors(
        error_df.filter(pl.col("vehicle_id").is_not_null()), smoothed_pred_col_func, []
    )
    grouped_smooth_error_df = group_and_aggregate(
        smoothed_error_df.filter(pl.col("vehicle_id").is_not_null()),
        "smooth",
        "merged_trajectory_length",
        "vehicle_id",
        groupby_cols=["sequence_id"],
    )

    # Usage example for raw data with custom prefix
    ci_pred_col_func = lambda col_suffix: f"ci_{col_suffix}"  # noqa: E731
    ci_error_df = calculate_errors(
        error_df.filter(pl.col("vehicle_id").is_not_null()), ci_pred_col_func, []
    )
    grouped_ci_error_df = group_and_aggregate(
        ci_error_df,
        "ci",
        "merged_trajectory_length",
        "vehicle_id",
        groupby_cols=["sequence_id"],
    )

    # raw_error
    raw_pred_col_func = lambda col_suffix: f"{col_suffix}_raw"  # noqa: E731
    raw_error_df = calculate_errors(
        error_df.filter(pl.col("object_id").is_not_null()),
        raw_pred_col_func,
        ["object_id"],
    )
    grouped_raw_error_df = group_and_aggregate(
        raw_error_df,
        "raw",
        "raw_trajectory_length",
        "object_id",
        groupby_cols=["sequence_id"],
    )

    # IMM error
    imm_pred_col_func = lambda col_suffix: f"{col_suffix}_imm"  # noqa: E731
    imm_error_df = calculate_errors(
        error_df.filter(pl.col("object_id_imm").is_not_null()),
        imm_pred_col_func,
        ["object_id_imm"],
    )
    grouped_imm_error_df = group_and_aggregate(
        imm_error_df,
        "imm",
        "raw_trajectory_length",
        "object_id_imm",
        groupby_cols=["sequence_id"],
    )

    return (
        pl.concat(
            [
                grouped_smooth_error_df.with_columns(
                    pl.lit("smooth").alias("method"),
                    pl.col(pl.FLOAT_DTYPES).cast(pl.Float64),
                ),
                grouped_ci_error_df.with_columns(
                    pl.lit("ci").alias("method"),
                    pl.col(pl.FLOAT_DTYPES).cast(pl.Float64),
                ),
                grouped_raw_error_df.with_columns(
                    pl.lit("raw").alias("method"),
                    pl.col(pl.FLOAT_DTYPES).cast(pl.Float64),
                ),
                grouped_imm_error_df.with_columns(
                    pl.lit("imm").alias("method"),
                    pl.col(pl.FLOAT_DTYPES).cast(pl.Float64),
                ),
            ]
        )
        .drop(
            # "lane_index",
            cs.by_name("^s_velocity_.*$"),
            cs.by_name("^.*_ae$"),
            cs.by_name("^.*_pearsonr$"),
        )
        .drop(
            cs.by_name("^.*_mse$"),
            cs.by_name("^.*sequences.*$"),
            "n_vehicles",
            "n_vehicles_per_sequence",
        )
        .with_columns(
            pl.col("average_coverage_percent") * 100,
        )
        .rename(
            {
                "s_rmse": "centroid_s_rmse",
                "speed_rmse": "Speed_speed_rmse",
                "d_rmse": "centroid_d_rmse",
            }
        )
        .group_by(groupby_col)
        .agg(pl.all().mean())
    )
