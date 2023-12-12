import os
from typing import Dict, List, Set, Tuple, TYPE_CHECKING
import numpy as np
import polars as pl
import h3.api.numpy_int as h3
import yaml
import utm
from shapely.geometry import Polygon
import math

from src.frenet import SplineLane


if TYPE_CHECKING:
    import plotly.graph_objects as go
    import geopandas as gpd


def safe_collect(df: pl.LazyFrame) -> pl.DataFrame:
    return df.collect() if isinstance(df, pl.LazyFrame) else df


def safe_lazy(df: pl.DataFrame) -> pl.LazyFrame:
    return df.lazy() if isinstance(df, pl.DataFrame) else df


# create a wrapper function to time the function
def timeit(func):
    def timed(*args, **kwargs):
        import time

        br = args[0]

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        if br.VERBOSITY > 0:
            print(
                "function: {} took: {} seconds".format(
                    func.__name__, end_time - start_time
                )
            )

        return result

    return timed


class BasicRadar:
    VERBOSITY = 1

    def __init__(
        self,
    ) -> None:
        self._crop_angle = 0.5 * 110 * (math.pi / 180)
        self.crop_radius_m = 250

    @classmethod
    @timeit
    def crop_angle(
        cls,
        df: pl.LazyFrame,
    ) -> pl.LazyFrame:
        crop_angle = cls._crop_angle
        return df.filter(
            pl.col("f32_positionX_m").abs()
            < (pl.col("f32_positionY_m") * np.tan(crop_angle))
        )

    @classmethod
    @timeit
    def create_object_id(cls, df: pl.DataFrame) -> pl.DataFrame:
        # make the object id the ui32_objectId + the ip + the date
        return df.with_columns(
            [
                pl.struct(
                    [
                        pl.col("ui32_objectID").cast(pl.Utf8),
                        pl.col("ip").cast(pl.Utf8),
                        pl.col("epoch_time").dt.strftime("%Y-%m-%d"),
                    ],
                )
                .hash()
                .alias("object_id"),
            ]
        )

    @classmethod
    @timeit
    def set_timezone(cls, df: pl.DataFrame, timezone_: str = "UTC") -> pl.DataFrame:
        return df.with_columns(
            [
                pl.col("epoch_time")
                .dt.replace_time_zone(timezone_)
                .alias("epoch_time"),
            ]
        )

    @classmethod
    @timeit
    def add_cst_timezone(
        cls,
        df: pl.DataFrame,
        time_col: str = "epoch_time",
        cst_col: str = "epoch_time_cst",
    ) -> pl.DataFrame:
        return df.with_columns(
            [
                pl.col(time_col)
                .dt.convert_time_zone("US/Central")
                .alias(cst_col)
            ]
        )

    @classmethod
    @timeit
    def fix_dst(cls, df: pl.DataFrame) -> pl.DataFrame:
        from datetime import timedelta

        return df.with_columns(
            [(pl.col("epoch_time_cst") - timedelta(hours=1)).alias("epoch_time_cst")]
        )

    @classmethod
    @timeit
    def crop_radius(cls, df: pl.DataFrame, radius: float = None) -> pl.DataFrame:
        radius = radius or cls.crop_radius_m
        return df.filter(
            (pl.col("f32_positionX_m") ** 2 + pl.col("f32_positionY_m") ** 2)
            <= (radius**2)
        )

    @classmethod
    @timeit
    def fix_duplicate_positions(cls, df: pl.DataFrame) -> pl.DataFrame:
        # get all float columns
        # float_cols = df.select(pl.col(pl.FLOAT_DTYPES).first()).columns

        return (
            df.sort(["object_id", "epoch_time"])
            .with_columns(
                [
                    (pl.col("ui16_ageCount").diff() == 0)
                    .over("object_id")
                    .alias("duplicate")
                ]
            )
            .filter(~pl.col("duplicate"))
            .drop("duplicate")
        )

    @classmethod
    @timeit
    def fix_stop_param_walk(cls, df: pl.DataFrame) -> pl.DataFrame:
        ffill_cols = {
            "f32_velocityInDir_mps",
            "utm_x",
            "utm_y",
            "lat",
            "lon",
            "direction",
        }.intersection(set(df.columns))

        return (
            df.sort("epoch_time")
            .with_columns(
                [
                    (
                        # mark stops (velocity < 0.01 m/s)
                        # & last velocity also < 0.01 m/s
                        (
                            (pl.col("f32_velocityInDir_mps").shift(1) < 0.01)
                            & (pl.col("f32_velocityInDir_mps") < 0.01)
                        ).fill_null(False)
                    )
                    .over("object_id")
                    .alias("stopped"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("stopped"))
                    .then(pl.lit(None))
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in ffill_cols
                ]
            )
            # forward fill the nulls
            .with_columns(
                [pl.col(c).forward_fill().over("object_id") for c in ffill_cols]
            )
            .drop(
                [
                    "stopped",
                ]
            )
        )

    @classmethod
    @timeit
    def filter_short_trajectories(
        cls,
        df: pl.DataFrame,
        minimum_distance_m: int = 200,
        minimum_duration_s: int = 5,
    ) -> pl.DataFrame:
        return (
            df.with_columns(
                [
                    (
                        (pl.col("utm_x").first() - pl.col("utm_x").last()).pow(2)
                        + (pl.col("utm_y").first() - pl.col("utm_y").last()).pow(2)
                    )
                    .sqrt()
                    .over("object_id")
                    .alias("straight_distance"),
                    (pl.col("epoch_time").last() - pl.col("epoch_time").first())
                    .over("object_id")
                    .dt.seconds()
                    .alias("duration"),
                ]
            )
            .filter(
                (pl.col("straight_distance") >= minimum_distance_m)
                & (pl.col("duration") >= minimum_duration_s)
            )
            .drop(["straight_distance", "duration"])
        )

    @classmethod
    @timeit
    def resample(
        cls, df: pl.DataFrame, resample_interval_ms: int = 100
    ) -> pl.DataFrame:
        assert resample_interval_ms > 0, "resample_interval_ms must be positive"
        assert "object_id" in df.columns, "object_id must be in the dataframe"
        return (
            df.sort(["object_id", "epoch_time"])
            # .pipe(safe_collect)
            .groupby_dynamic(
                index_column="epoch_time",
                every=f"{resample_interval_ms}ms",
                by=["object_id"],
                check_sorted=False,
            )
            .agg(
                [
                    pl.col(pl.FLOAT_DTYPES).mean(),
                    *(
                        pl.col(type_).first()
                        for type_ in [pl.INTEGER_DTYPES, pl.Utf8, pl.Boolean]
                    ),
                ]
            )
        )

    @classmethod
    @timeit
    def rad_to_degrees(
        cls,
        df: pl.DataFrame,
        rad_col: str = "direction",
        out_col: str = "direction_degrees",
    ) -> pl.DataFrame:
        return df.with_columns([(pl.col(rad_col) * (180 / np.pi)).alias(out_col)])

    @classmethod
    @timeit
    def to_geodataframe(cls, df: pl.DataFrame) -> "gpd.GeoDataFrame":
        import geopandas as gpd

        return gpd.GeoDataFrame(
            df.to_pandas(use_pyarrow_extension_array=False),
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326",
        )

    @classmethod
    @timeit
    def add_lane_group(cls, df: pl.DataFrame, lane_col: str = "lane") -> pl.DataFrame:
        collect = True
        if isinstance(df, pl.LazyFrame):
            # don't collect the dataframe
            collect = False
        return (
            df.lazy()
            .sort(["epoch_time"])
            .set_sorted(["epoch_time"])
            .with_columns(
                # calculate the time delta
                [
                    (
                        (pl.col(lane_col) != pl.col(lane_col).shift(1))
                        | (
                            pl.col(lane_col).is_null()
                            & pl.col(lane_col).shift(1).is_null()
                        )
                    )
                    .fill_null(False)
                    .over("object_id")
                    .alias("lane_change"),
                ]
            )
            .with_columns(
                [
                    # identify gaps in the time series larger than 0.2
                    (
                        pl.col("lane_change")
                        .cast(pl.Int32)
                        .cumsum()
                        .over(
                            [
                                "object_id",
                            ]
                        )
                        .alias("lane_group")
                    )
                ]
            )
            .pipe(safe_collect if collect else lambda x: x)
        )

    @classmethod
    @timeit
    def new_object_id(
        cls,
        df: pl.DataFrame,
        object_id_cols: Tuple[str] = ("object_id", "lane_group"),
    ) -> pl.DataFrame:
        assert all(c in df.columns for c in object_id_cols), (
            "object_id_cols must be in the dataframe",
        )

        return df.with_columns(
            [
                pl.concat_str(
                    [pl.col(c).cast(pl.Utf8) for c in object_id_cols], separator="~"
                )
                .cast(pl.Categorical)
                .cast(pl.Int64)
                .alias("object_id"),
            ]
        )

    @classmethod
    @timeit
    def rotate_radars(
        cls, df: pl.DataFrame, angle_mapping: Dict[str, float] = None
    ) -> pl.DataFrame:
        df = df.with_columns(
            [pl.col("ip").map_dict(angle_mapping).alias("rotation_angle")]
        ).with_columns(
            [
                (
                    pl.col("f32_positionX_m") * pl.col("rotation_angle").cos()
                    - pl.col("f32_positionY_m") * pl.col("rotation_angle").sin()
                ).alias("rotated_x"),
                (
                    pl.col("f32_positionX_m") * pl.col("rotation_angle").sin()
                    + pl.col("f32_positionY_m") * pl.col("rotation_angle").cos()
                ).alias("rotated_y"),
            ]
        )

        if "heading" in df.columns:
            df = df.with_columns(
                (pl.col("rotation_angle") + pl.col("heading")).alias("heading_utm")
            )

        return df.drop(["rotation_angle"])

    @classmethod
    @timeit
    def update_origin(
        cls, df: pl.DataFrame, origin_dict: Dict[str, Tuple[float]]
    ) -> pl.DataFrame:
        # split into two dictionaries, one with the x and one with the y
        x_dict = {k: v[0] for k, v in origin_dict.items()}
        y_dict = {k: v[1] for k, v in origin_dict.items()}

        return (
            df.with_columns(
                [
                    pl.col("ip").map_dict(x_dict).alias("x_origin"),
                    pl.col("ip").map_dict(y_dict).alias("y_origin"),
                ]
            )
            .with_columns(
                [
                    (pl.col("rotated_x") + pl.col("x_origin")).alias("utm_x"),
                    (pl.col("rotated_y") + pl.col("y_origin")).alias("utm_y"),
                ]
            )
            .drop(["x_origin", "y_origin"])
        )

    @classmethod
    @timeit
    def _get_directed_d(
        cls,
        df: pl.DataFrame,
        x1_col: str = "utm_x",
        y1_col: str = "utm_y",
        x2_col: str = "x",
        y2_col: str = "y",
        d_col: str = "min_d",
        ds_du_x_col: str = "ds_du_x",
        ds_du_y_col: str = "ds_du_y",
    ) -> pl.DataFrame:
        # calculate the directives
        # https://stackoverflow.com/a/16544330
        # dot = x1*x2 + y1*y2      # Dot product between [x1, y1] and [x2, y2]
        # det = x1*y2 - y1*x2      # Determinant
        return (
            df.with_columns(
                [
                    (pl.col(x2_col) - pl.col(x1_col)).alias("dx"),
                    (pl.col(y2_col) - pl.col(y1_col)).alias("dy"),
                ]
            )
            .with_columns(
                [
                    # calculate the angle between the lane and the car
                    (
                        # calculate the dot product
                        pl.col("dx") * pl.col(ds_du_x_col)
                        + pl.col("dy") * pl.col(ds_du_y_col)
                    ).alias("dot"),
                    # calculate the cross product
                    (
                        pl.col("dx") * pl.col(ds_du_y_col)
                        - pl.col("dy") * pl.col(ds_du_x_col)
                    ).alias("cross"),
                ]
            )
            .with_columns(pl.atan2(pl.col("cross"), pl.col("dot")).alias("angle_d"))
            .with_columns(
                [
                    pl.when(pl.col("angle_d") > np.pi)
                    .then(pl.col(d_col) * -1)
                    .otherwise(pl.col(d_col))
                    .alias(d_col),
                ]
            )
            .drop(
                [
                    "dot",
                    "cross",
                    "angle_d",
                    "dx",
                    "dy",
                ]
            )
        )

    @classmethod
    @timeit
    def map_frenet_lanes(
        cls,
        df: pl.DataFrame,
        lanes: List[SplineLane],
        distance_threshold: float = 3,
        drop_outside: bool = True,
        directed: bool = True,
    ) -> pl.DataFrame:
        """
        *This will drop any data that is greater than distance_threshold from a lane*

        Args:
            df (pl.DataFrame): _description_
            lanes (List[SplineLane]): _description_
            distance_threshold (float, optional): _description_. Defaults to 6.

        Returns:
            pl.DataFrame: _description_
        """

        from scipy.spatial import KDTree

        # create a dataframe of all the lanes
        kdtree_df = pl.concat(
            [
                lane.fitted_pl_df.with_columns(
                    [
                        pl.lit(lane.name).alias("lane"),
                    ]
                )
                for lane in lanes
            ]
        ).with_row_count()

        # create a kdtree
        kdtree = KDTree(kdtree_df[["x", "y"]].to_numpy())

        # get the nearest lane
        d, nearest_lane_index = kdtree.query(
            df[["utm_x", "utm_y"]].to_numpy(),
            k=1,
            distance_upper_bound=distance_threshold,
            p=2,
            workers=-1,
            eps=1e-9,
        )

        # replace the inf values with nan
        df = (
            df.with_columns(
                [
                    pl.Series(
                        "lane_ind",
                        nearest_lane_index,
                        dtype=pl.UInt32,
                    ),
                    pl.Series(
                        "min_d",
                        d,
                        dtype=pl.Float32,
                    ),
                ]
            )
            # get the nearest lane
            .join(
                kdtree_df,
                how="inner" if drop_outside else "left",
                left_on="lane_ind",
                right_on="row_nr",
            )
        )
        return (
            df.pipe(
                cls._get_directed_d,
            )
            if directed
            else df
        )

    @classmethod
    @timeit
    def map_frenet_lane_constrained(
        cls,
        df: pl.DataFrame,
        lanes: List[SplineLane],
        filter_expr: pl.Expr,
        distance_threshold: float = 6,
        lane_col: str = "lane",
    ) -> pl.DataFrame:
        if "row_nr" not in df.columns:
            df = df.with_row_count()

        snap_df = df.filter(filter_expr)

        new_dfs = []
        for l in lanes:
            l_df = snap_df.filter((pl.col(lane_col) == l.name))
            # create a dataframe of all the lanes
            d, ind = l.kdtree.query(
                l_df[["utm_x", "utm_y"]].to_numpy(),
                p=2,
                workers=-1,
                distance_upper_bound=distance_threshold,
            )
            ltree_df = l.fitted_pl_df
            l_df = (
                l_df.with_columns(
                    [
                        pl.Series(
                            "lane_ind",
                            ind,
                            dtype=pl.UInt32,
                        ),
                        pl.Series(
                            "min_d",
                            d,
                            dtype=pl.Float64,
                        ),
                    ]
                )
                .drop(ltree_df.columns)
                .join(
                    ltree_df.with_row_count(),
                    how="inner",
                    left_on="lane_ind",
                    right_on="row_nr",
                )
                .pipe(
                    cls._get_directed_d,
                )
            )

            new_dfs.append(l_df)

        if not new_dfs:
            return df

        new_df = pl.concat(new_dfs)

        return df.update(
            new_df,
            how="left",
            on=["row_nr"],
        )

    @classmethod
    @timeit
    def get_s_frame_lateral_error_df(
        cls, df: pl.DataFrame, agg_interval: int = 1, normalize: bool = True
    ) -> pl.DataFrame:
        assert (
            "s" in df.columns
        ), "s must be in the dataframe. Call map_frenet_lanes first"
        assert (
            "min_d" in df.columns
        ), "min_d must be in the dataframe. Call map_frenet_lanes first"

        if agg_interval > 1:
            raise NotImplementedError("agg_interval > 1 not implemented yet")

        return (
            df.lazy()
            .with_columns(
                [
                    # cast s to int
                    pl.col("s").round().cast(pl.Int32).alias("s_int"),
                ]
            )
            .groupby(["ip", "lane", "s_int"])
            .agg(
                [
                    pl.col("min_d").mean().alias("min_d"),
                    pl.col("min_d").quantile(0.25).alias("min_d_25"),
                    pl.col("min_d").quantile(0.75).alias("min_d_75"),
                ]
            )
            .sort(by=["ip", "s_int"])
            .collect()
        )

    @classmethod
    @timeit
    def norm_s_to_stopbar(
        cls, df: pl.DataFrame, lanes: List[SplineLane]
    ) -> pl.DataFrame:
        assert (
            "tl" in df.columns
        ), "tl must be in the dataframe. This should map a radar to the traffic signal"
        assert (
            "s_int" in df.columns
        ), "s_int must be in the dataframe. Call get_s_frame_lateral_error_df first"

        local_df = df.lazy().with_columns([pl.lit(0).alias("s_int_norm")])

        for l in lanes:
            for detector in l.detectors:
                local_df = local_df.with_columns(
                    pl.when(
                        (pl.col("lane") == l.name)
                        & (pl.col("tl") == detector.attrs["tl"])
                    )
                    .then((pl.col("s_int") - detector.s))
                    .otherwise(pl.col("s_int_norm"))
                    .alias("s_int_norm")
                )

        local_df = local_df.collect()

        # reverse the sign of subtraction for radars that point in the opposite direction
        reverse = (
            local_df.groupby(["ip", "lane"])
            .agg([pl.col("s_int_norm").mean().alias("s_int_norm")])
            .filter(pl.col("s_int_norm") < 0)
            .select(["ip", "lane"])
        )

        return local_df.with_columns(
            [
                pl.when(
                    pl.col("ip").is_in(reverse["ip"])
                    & pl.col("lane").is_in(reverse["lane"])
                )
                .then(pl.col("s_int_norm") * -1)
                .otherwise(pl.col("s_int_norm"))
                .alias("s_int_norm")
            ]
        )

    @timeit
    def correct_center(
        cls,
        df: pl.DataFrame,
        x_col: str = "f32_positionX_m",
        y_col: str = "f32_positionY_m",
    ) -> pl.DataFrame:
        return df.with_columns(
            [
                # precompute the angle sin and cos
                (
                    pl.col(x_col)
                    - (pl.col("f32_directionX") * pl.col("f32_distanceToBack_m"))
                ).alias(x_col),
                (
                    pl.col(y_col)
                    - (pl.col("f32_directionY") * pl.col("f32_distanceToBack_m"))
                ).alias(y_col),
            ]
        )

    @classmethod
    @timeit
    def add_heading(
        cls,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        return df.with_columns(
            pl.arctan2(pl.col("f32_directionY"), pl.col("f32_directionX")).alias(
                "heading"
            )
        )

    @classmethod
    @timeit
    def clip_trajectory_end(
        cls,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        return (
            df.with_columns(
                [
                    # Trim off the ends of trajectoriers where the radar has done predictions.
                    # This can be done by reversing the cummulative count of predictions and comparing to the reverse cummulative count
                    # They are equal until the last measure data point
                    (
                        pl.col("ui16_predictionCount").count()
                        - pl.col("ui16_predictionCount").cumcount()
                    )
                    .over("object_id")
                    .alias("cumcount"),
                    (pl.col("ui16_predictionCount") > 0)
                    .reverse()
                    .cumsum()
                    .reverse()
                    .over("object_id")
                    .alias("reverse_cumcount"),
                ]
            )
            .with_columns(
                [
                    (pl.col("cumcount") != pl.col("reverse_cumcount")).alias("keep"),
                ]
            )
            .filter(pl.col("keep"))
            .drop(["keep", "cumcount", "reverse_cumcount"])
        )


class CalibratedRadar(BasicRadar):
    def __init__(
        self,
        radar_location_path: str,
        # overlap_zone_path: str,
    ) -> None:
        super().__init__()

        self.h3_resolution = 14

        (
            self.radar_locations,
            self.utm_zone,
            self.rotations,
        ) = self._read_radar_locations(radar_location_path)

    @staticmethod
    def _read_radar_locations(path) -> Tuple[dict, Tuple]:
        with open(path, "r") as f:
            radar_locations = yaml.safe_load(f)

        def _convert_latlon(origin_tuple):
            try:
                return utm.from_latlon(*origin_tuple)
            except Exception:
                return utm.from_latlon(*origin_tuple[::-1])

        radar_utms = {
            ip: _convert_latlon(radar_locations[ip]["origin"]) for ip in radar_locations
        }

        radar_rotations = {
            ip: radar_locations[ip]["angle"] * (math.pi / 180) for ip in radar_locations
        }

        # assert that all radar utm zones are the same
        assert len({(x[2], x[3]) for x in radar_utms.values()}) == 1

        return radar_utms, list(radar_utms.values())[0][2:4], radar_rotations

    @timeit
    def rotate_heading(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            # do this faster
            df.lazy()
            .with_columns(
                pl.atan2(pl.col("f32_directionX"), pl.col("f32_directionY")).alias(
                    "arctan2"
                ),
                pl.col("ip").map_dict(self.rotations).alias("rotation"),
            )
            .with_columns(
                [
                    (
                        (pl.col("arctan2") + pl.col("rotation") + np.pi) % (2 * np.pi)
                    ).alias("direction"),
                ]
            )
            .pipe(self.rad_to_degrees)
            .drop(["arctan2", "rotation"])
            .collect()
        )

    def rotate_radars(
        self,
        df: pl.DataFrame,
        rotations: Dict[str, float] = None,
    ) -> pl.DataFrame:
        rotations = rotations or self.rotations
        return df.pipe(
            super().rotate_radars,
            rotations,
        )

    def update_origin(
        self, df: pl.DataFrame, locations: Dict[str, Tuple[float]] = None
    ) -> pl.DataFrame:
        locations = locations or self.radar_locations
        return df.pipe(super().update_origin, locations)

    @timeit
    def radar_to_latlon(
        self,
        df: pl.DataFrame,
        utm_x_col: str = "utm_x",
        utm_y_col: str = "utm_y",
        lat_col: str = "lat",
        lon_col: str = "lon",
    ) -> pl.DataFrame:
        # add a row number column

        df = safe_collect(df)

        # convert to latlon using utm. This could be chunked
        # TODO: chunk this
        x, y = utm.to_latlon(
            df[utm_x_col].to_numpy(),
            df[utm_y_col].to_numpy(),
            self.utm_zone[0],
            self.utm_zone[1],
        )

        # convert back to polars
        return df.drop([c for c in [lat_col, lon_col] if c in df.columns]).with_columns(
            [
                pl.Series(lat_col, x, dtype=pl.Float64),
                pl.Series(lon_col, y, dtype=pl.Float64),
            ]
        )

    @timeit
    def radar_to_h3(
        self, df: pl.DataFrame, col_name: str = "h3", resolution: int = None
    ) -> pl.DataFrame:
        if resolution is None:
            resolution = self.h3_resolution
        return df.with_column(
            pl.struct(["lat", "lon"])
            .apply(
                lambda x: h3.geo_to_h3(x["lat"], x["lon"], resolution),
                return_dtype=pl.Utf8,
            )
            .alias(col_name)
        )

    @timeit
    def int_h3_2_str(self, df: pl.DataFrame) -> pl.DataFrame:
        if df["h3"].dtype == pl.Utf8:
            return df

        import h3.api.basic_int as h3_int

        return df.with_columns([pl.col("h3").apply(h3_int.h3_to_string).alias("h3")])

    @timeit
    def filter_network_boundaries(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("h3").is_in(list(self.network_boundary)))


def plot_lateral_error(
    error_df: pl.DataFrame,
) -> "go.Figure":
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # convert to a dataframe

    # create a coloring for the lanes
    lane_color_map = dict(zip(error_df["lane"].unique(), px.colors.qualitative.D3))

    fig = make_subplots(
        rows=error_df["ip"].n_unique(),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.05,
        subplot_titles=error_df["ip"].unique(),
    )

    for i, (ip, ip_df) in enumerate(error_df.groupby("ip")):
        for lane, lane_df in ip_df.groupby("lane"):
            fig.add_trace(
                go.Scatter(
                    x=lane_df["s_int_norm"],
                    y=lane_df["min_d"],
                    mode="lines",
                    name=lane,
                    line=dict(color=lane_color_map[lane]),
                ),
                row=i + 1,
                col=1,
            )

            # add the error band
            fig.add_trace(
                go.Scatter(
                    x=lane_df["s_int_norm"],
                    y=lane_df["min_d_25"],
                    mode="lines",
                    name=lane,
                    showlegend=False,
                    line=dict(color=lane_color_map[lane]),
                ),
                row=i + 1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=lane_df["s_int_norm"],
                    y=lane_df["min_d_75"],
                    mode="lines",
                    name=lane,
                    showlegend=False,
                    fill="tonexty",
                    line=dict(color=lane_color_map[lane]),
                ),
                row=i + 1,
                col=1,
            )

            # update the height
            fig.update_layout(
                height=300 * error_df["ip"].n_unique(),
                showlegend=False,
                title_text="Lateral Error",
                yaxis_title="Lateral Error (m)",
                xaxis_title="Distance to Stopbar (m)",
                **{
                    f"xaxis{i+1}": dict(
                        title="Distance to Stopbar (m)",
                    )
                    for i in range(error_df["ip"].n_unique() - 1)
                },
            )

            # update the x axis

    return fig


def plot_trajectories(
    df: pl.DataFrame,
    color_col: str = None,
    plotly: bool = False,
) -> None:
    if plotly:
        raise NotImplementedError("plotly not implemented yet")

    import matplotlib.pyplot as plt
    import contextily as ctx

    fig, ax = plt.subplots(figsize=(10, 10))

    master_gdf = BasicRadar.to_geodataframe(df).to_crs(epsg=3857)
    master_gdf["x"] = master_gdf.geometry.x
    master_gdf["y"] = master_gdf.geometry.y

    bounds = master_gdf.total_bounds

    # add 50 meters to the bounds
    bounds = (
        bounds[0] - 50,
        bounds[1] - 50,
        bounds[2] + 50,
        bounds[3] + 50,
    )

    # set the bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    master_gdf.drop(["geometry"], axis=1, inplace=True)

    for _, veh_df in master_gdf.groupby("object_id"):
        color = "black" if color_col is None else veh_df[color_col].iloc[0]

        veh_df.plot(
            ax=ax,
            x="x",
            y="y",
            color=color,
            linewidth=4 if color == "black" else 2,
            markevery=[-1],
            markersize=0,
            marker="o",
        )

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, attribution="")

    # turn off legend
    ax.legend().remove()
    plt.axis("off")

    return fig, ax


def save_images_every_01_seconds(
    df: pl.DataFrame,
    output_folder: str,
    image_format: str = "png",
    color_col: str = None,
    time_step: float = 0.1,
    show_trace: bool = False,
    dry_run: bool = False,
) -> None:
    from datetime import timedelta
    import pathlib
    import matplotlib.pyplot as plt
    import contextily as ctx

    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    time_step = timedelta(seconds=time_step)

    master_gdf = BasicRadar.to_geodataframe(df)

    master_gdf = master_gdf.to_crs(epsg=3857)

    master_gdf["x"] = master_gdf.geometry.x
    master_gdf["y"] = master_gdf.geometry.y

    bounds = master_gdf.total_bounds

    min_time = master_gdf["epoch_time"].min()
    max_time = master_gdf["epoch_time"].max()
    image_index = 0
    current_time = min_time
    while current_time <= max_time:
        filtered_df = (
            master_gdf.loc[
                master_gdf["epoch_time"].between(min_time, current_time + time_step)
            ]
            .copy()
            .drop(["geometry"], axis=1)
        )

        fig, ax = plt.subplots(figsize=(10, 10))
        # lanes_gdf.plot(ax=ax, color="red", linewidth=6)

        for _, veh_df in filtered_df.groupby("object_id"):
            #
            if show_trace:
                veh_df.plot(
                    ax=ax,
                    x="x",
                    y="y",
                    color="black" if color_col is None else veh_df[color_col].iloc[0],
                    linewidth=3,
                    markevery=[-1],
                    markersize=3,
                    marker="o",
                    # zorder=2,
                )

            else:
                veh_df.iloc[-1:].plot(
                    ax=ax,
                    x="x",
                    y="y",
                    color="black" if color_col is None else veh_df[color_col].iloc[0],
                    linewidth=0,
                    markersize=10,
                    marker="o",
                    # zorder=2,
                )

        # turn off legend
        ax.legend().remove()
        # Set the limits for x and y axes to focus on the area of interest
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])

        plt.axis("off")

        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, attribution="")

        if dry_run:
            return fig, ax
        # update the zoom level

        plt.savefig(
            os.path.join(output_folder, f"image_{image_index:04d}.{image_format}"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

        min_time = min_time if show_trace else min_time + time_step
        current_time += time_step
        image_index += 1
