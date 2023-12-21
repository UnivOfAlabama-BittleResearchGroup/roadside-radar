from functools import lru_cache
import math
from typing import List, Union
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
from shapely import affinity
from shapely.ops import split
from shapelysmooth import chaikin_smooth, taubin_smooth
import polars as pl
import geopandas as gpd
import json


def convert_2_utm(pos: Union[np.ndarray, tuple]) -> np.ndarray:
    import utm

    if isinstance(pos, tuple):
        pos = np.array(pos).reshape(1, -1)

    x, y, _, _ = utm.from_latlon(pos[:, 0], pos[:, 1])

    return np.stack([x, y], axis=1)


def linestring_to_points(linestring: LineString) -> List[Point]:
    return [Point(x, y) for x, y in linestring.coords]


def load_centerlines(
    path_2_centerlines: str, explode_lines: bool = True
) -> gpd.GeoDataFrame:
    lane_df = gpd.read_file(path_2_centerlines)

    if explode_lines:
        lane_df = explode_linestring(lane_df, explode_col="points").reset_index(
            drop=True
        )

        # convert to utm
        lane_df = lane_df.to_crs(lane_df.estimate_utm_crs())

    return lane_df


def get_radar_origins(path_2_origin_angles: str) -> gpd.GeoDataFrame:
    with open(path_2_origin_angles) as f:
        radar_info = pd.DataFrame(json.load(f)).T

    radar_info[["lon", "lat"]] = pd.DataFrame(
        radar_info["origin"].tolist(), index=radar_info.index
    )

    radar_info = gpd.GeoDataFrame(
        radar_info,
        geometry=gpd.points_from_xy(radar_info.lon, radar_info.lat),
        crs="EPSG:4326",
    )

    # convert to utm
    radar_info = radar_info.to_crs(radar_info.estimate_utm_crs())
    return radar_info


def get_radar_fov(
    path_2_origin_angles: str, radius: Union[float, dict] = 250.0, angle: float = 110.0
) -> gpd.GeoDataFrame:
    # read in the radar origin json file
    radar_info = get_radar_origins(path_2_origin_angles)
    utm_crs = radar_info.crs

    if isinstance(radius, float):
        radar_info["radius"] = radius
    else:
        radar_info["radius"] = radar_info.index.map(radius)

    # create a circle around the radar origin
    radar_info["fov"] = radar_info.geometry.buffer(radar_info["radius"])

    # draw a line from the center of the radar to the edge of the fov
    radar_info["fov_line"] = radar_info.apply(
        lambda x: LineString([x.geometry, (x.geometry.x, x.geometry.y + x.radius + 1)]),
        axis=1,
    )

    # rotate the fov line to boresight of the radar
    radar_info["fov_line"] = radar_info.apply(
        lambda x: affinity.rotate(x.fov_line, x.angle, origin=x.geometry), axis=1
    )

    # add in the line string of +- fov angle / 2
    radar_info["fov_line"] = radar_info.apply(
        lambda x: LineString(
            [
                *affinity.rotate(x.fov_line, angle / 2, origin=x.geometry).coords,
                *affinity.rotate(x.fov_line, angle / -2, origin=x.geometry).coords,
            ]
        ),
        axis=1,
    )

    # split fov circle with fov line
    radar_info["fov"] = radar_info.apply(
        lambda x: sorted(
            list(split(x.fov, x.fov_line).geoms),
            key=lambda y: y.area if y.area > 1 else 1e9,
        )[0],
        axis=1,
    )

    return radar_info.set_geometry("fov").set_crs(utm_crs).to_crs("EPSG:4326")


def explode_linestring(
    df: gpd.GeoDataFrame, explode_col: str = "points"
) -> gpd.GeoDataFrame:
    df[explode_col] = df.geometry.apply(lambda x: linestring_to_points(x))
    # explode the points
    return (
        df.explode(explode_col, ignore_index=False, index_parts=True)
        .set_geometry(explode_col)
        .drop(columns=["geometry"])
        .set_crs(df.crs)
    )


def redistribute_vertices(geom, distance):
    # from https://gis.stackexchange.com/a/367965
    if geom.geom_type == "LineString":
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [
                geom.interpolate(float(n) / num_vert, normalized=True)
                for n in range(num_vert + 1)
            ]
        )
    elif geom.geom_type == "MultiLineString":
        parts = [redistribute_vertices(part, distance) for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError("unhandled geometry %s", (geom.geom_type,))


class Lane:
    def __init__(
        self,
        name: str,
        linestring: LineString,
        utm_crs: str,
        crs: str,
    ) -> None:
        self.name = name
        self.linestring = linestring
        self._smoothed_linestring = None
        self.utm_crs = utm_crs
        self.crs = crs
        self._fitted = False
        self._fitted_pl_df = None
        self._step_size = None

    def __repr__(self) -> str:
        return f"Lane({self.name})"

    def __str__(self) -> str:
        return f"Lane({self.name})"

    def __eq__(self, o: object) -> bool:
        return self.name == o.name

    def __hash__(self) -> int:
        return hash(self.name)

    def fit(self, step_size: float = 0.1) -> "Lane":
        smoothed_linestring = chaikin_smooth(
            # taubin_smooth(
            self.linestring,
            # )
        )
        self._smoothed_linestring = redistribute_vertices(
            smoothed_linestring, step_size
        )
        self._fitted = True
        self._step_size = step_size

        return self

    def frenet2xy(self, df: pl.DataFrame, s_col: str, d_col: str) -> pl.DataFrame:
        nearest_df = (
            df.with_columns(pl.col(s_col).cast(float))
            .sort(s_col)
            .join_asof(
                self.df.sort("s"),
                right_on="s",
                left_on=s_col,
                tolerance=self._step_size,
                suffix="_lane",
            )
        )

        # calculate the x and y position of the vehicle
        nearest_df = nearest_df.with_columns(
            x_lane=pl.col("x_lane") + pl.col(d_col) * pl.col("angle").sin(),
            y_lane=pl.col("y_lane") + pl.col(d_col) * pl.col("angle").cos(),
        )

        return nearest_df

    @property
    def df(self) -> pl.DataFrame:
        if self._fitted_pl_df is None:
            assert self._fitted, "Must fit the lane before accessing the dataframe"

            angles = np.arctan2(
                np.diff(self._smoothed_linestring.xy[1]),
                np.diff(self._smoothed_linestring.xy[0]),
            )

            # calculate the distance between points
            distances = np.sqrt(
                np.diff(self._smoothed_linestring.xy[0]) ** 2
                + np.diff(self._smoothed_linestring.xy[1]) ** 2
            )

            self._fitted_pl_df = pl.DataFrame(
                {
                    "x": np.array(self._smoothed_linestring.xy[0][:-1]),
                    "y": np.array(self._smoothed_linestring.xy[1][:-1]),
                    "s": np.cumsum(distances),
                    "angle": angles,
                }
            )

        return self._fitted_pl_df

    def get_intersection(self, other: LineString) -> Point:
        int_point = self.linestring.intersection(other)

        if int_point.is_empty:
            return None

        return self.df.sort(
            (pl.col("x") - int_point.x) ** 2 + (pl.col("y") - int_point.y) ** 2
        )[0].clone()


class RoadNetwork:
    def __init__(
        self,
        lane_gdf: gpd.GeoDataFrame,
        keep_lanes: List[str] = None,
        step_size: float = 0.1,
    ) -> None:
        assert "name" in lane_gdf.columns, "lane_gdf must have a 'name' column"

        if keep_lanes is None:
            keep_lanes = lane_gdf["name"].unique().tolist()

        self.lane_gdf = (
            lane_gdf.query("name in @keep_lanes")
            .reset_index(drop=True)
            .to_crs(lane_gdf.estimate_utm_crs())
            .explode(index_parts=False)
            .reset_index(drop=True)
        )

        self._utm_crs = self.lane_gdf.estimate_utm_crs()
        self._crs = self.lane_gdf.crs

        self._step_size = step_size

        self._lanes = self._build_lanes()

        self._pl_df = None
        self._tree = None

    @property
    def step_size(self, ) -> float:
        return self._step_size

    def _build_lanes(self) -> List[Lane]:
        return self.lane_gdf.apply(
            lambda row: Lane(
                name=row["name"],
                linestring=row["geometry"],
                crs=self._crs,
                utm_crs=self._utm_crs,
            ).fit(self._step_size),
            axis=1,
        ).to_list()

    def __repr__(self) -> str:
        return f"RoadNetwork({[lane.name for lane in self._lanes]})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def utm_crs(self) -> str:
        return self._utm_crs

    @property
    def df(self) -> pd.DataFrame:
        if self._pl_df is None:
            self._pl_df = pl.concat(
                lane.df.with_columns(
                    pl.lit(lane.name).alias("name"),
                )
                for lane in self._lanes
            ).with_row_count("lane_index")
        return self._pl_df

    def map_to_lane(
        self,
        df: pl.DataFrame,
        directed: bool = True,
        dist_upper_bound: float = 6,
        utm_x_col: str = "utm_x",
        utm_y_col: str = "utm_y",
        d_out_col: str = "d",
    ) -> pl.DataFrame:
        from scipy.spatial import KDTree

        # construct a kd tree
        if self._tree is None:
            self._tree = KDTree(
                np.stack(
                    [
                        self.df["x"].to_numpy(),
                        self.df["y"].to_numpy(),
                    ],
                    axis=1,
                )
            )

        # query the kd tree
        dist, idx = self._tree.query(
            df[[utm_x_col, utm_y_col]].to_numpy(),
            p=2,
            distance_upper_bound=dist_upper_bound,
        )

        # assign a direction to the distance (i.e positive or negative)
        df = (
            df.with_columns(
                pl.Series("lane_index", idx),
                pl.Series(d_out_col, dist),
            )
            .with_columns(
                pl.col("lane_index").cast(pl.UInt32),
            )
            .join(
                self.df.rename({"x": "x_lane", "y": "y_lane"}),
                on="lane_index",
                how="left",
                suffix="_lane",
            )
        )

        if directed:
            PI = math.pi
            TAU = 2 * math.pi
            PITAU = PI + TAU

            # calculate the clockwise angle between the lane and the vehicle
            df = df.with_columns(
                pl.when(
                    (
                        (
                            pl.arctan2(
                                pl.col(utm_y_col) - pl.col("y_lane"),
                                pl.col(utm_x_col) - pl.col("x_lane"),
                            )
                            - pl.col("angle")
                            + PITAU
                        )
                        % TAU
                        - PI
                    )
                    > 0
                )
                .then(pl.col(d_out_col))
                .otherwise(
                    pl.col(d_out_col) * -1,
                )
                .alias(d_out_col),
            )

        return df

    def frenet2xy(
        self, df: pl.DataFrame, lane_col: str, s_col: str, d_col: str
    ) -> pl.DataFrame:
        # get the lane index
        dfs = []
        for lane in self._lanes:
            dfs.append(
                df.filter(
                    pl.col(lane_col) == lane.name,
                ).pipe(
                    lane.frenet2xy,
                    s_col=s_col,
                    d_col=d_col,
                )
            )

        return pl.concat(dfs).sort(s_col)

    def get_intersections(self, other: LineString) -> Point:
        intersection_df = []
        for lane in self._lanes:
            intersection = lane.get_intersection(other)

            if intersection is not None:
                intersection_df.append(
                    intersection.with_columns(
                        pl.lit(lane.name).alias("lane"),
                    )
                )
        return pl.concat(intersection_df)
