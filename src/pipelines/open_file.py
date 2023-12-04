from pathlib import Path
import polars as pl


from src.radar import BasicRadar
from src.polars_utils import safe_collect


def open_file(file_path: Path) -> pl.DataFrame:
    return pl.scan_parquet(file_path)


def filter_func(df: pl.DataFrame, additional_filtering: pl.Expr = None) -> pl.DataFrame:
    return df if additional_filtering is None else df.filter(additional_filtering)


def process_df(
    df: pl.LazyFrame, f: BasicRadar, additional_filtering: pl.Expr = None
) -> pl.DataFrame:
    if not isinstance(df, pl.LazyFrame):
        df = df.lazy()

    df = (
        df
        # create the object_id column
        .pipe(f.create_object_id)
        # convert the object_id column to int
        .pipe(f.object_id_2_int)
        .pipe(f.correct_center)
        # sort by object_id and epoch_time
        .sort(by=["object_id", "epoch_time"])
        .set_sorted(["object_id", "epoch_time"])
        # filter out vehicles that don't trave some minimum distance (takes care of radar noise)
        .pipe(f.filter_short_trajectories, minimum_distance_m=10, minimum_duration_s=2)
        # resample to 10 Hz
        .pipe(f.resample, 100)
        # smooth the values during stop events. This is allowed because there is no
        # .pipe(f.fix_stop_param_walk)
        # # fix when the radar is outputs the same data for multiple frames
        .pipe(f.fix_duplicate_positions)
        # clip the end of trajectories where the velocity is constant
        # .pipe(f.clip_trajectory_end)
        .pipe(f.set_timezone, timezone_="UTC")
        .pipe(f.add_cst_timezone)
        # filter just the first 12 hours of data
        # .pipe(f.crop_radius, 400)
        .pipe(f.rotate_radars)
        .pipe(f.update_origin)
        .pipe(f.rotate_heading)
        .pipe(filter_func, additional_filtering=additional_filtering)
        .pipe(safe_collect)
        .pipe(f.radar_to_latlon)
        # TODO: Crop out when ui16_predictionCount is high.
        # We can do a better job of prediction
    )

    if "row_nr" not in df.columns:
        df = df.with_row_count()

    return (
        df.sort(by=["object_id", "epoch_time"])
        .set_sorted(["object_id", "epoch_time"])
        .rechunk()
    )
