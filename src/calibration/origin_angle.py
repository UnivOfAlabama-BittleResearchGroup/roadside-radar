# find the root of the project
from typing import Tuple
import numpy as np
import polars as pl
import geopandas as gpd
from src.geometry import redistribute_vertices, explode_linestring
from shapelysmooth import chaikin_smooth, taubin_smooth
from scipy.spatial import KDTree
from scipy.optimize import minimize



def build_lane_df(lane_center_path: str) -> Tuple[gpd.GeoDataFrame, pl.DataFrame]:
    lane_df = gpd.read_file(lane_center_path)
    # utm_crs = lane_df.estimate_utm_crs()

    # # for use later
    # lane_geometry = lane_df.copy()

    lane_df = (
        lane_df.to_crs(lane_df.estimate_utm_crs())
        .explode(index_parts=False)
        .reset_index(drop=True)
    )

    lane_df["geometry"] = (
        lane_df["geometry"]
        .apply(
            taubin_smooth,
            # args=(3,  True)
        )
        .apply(chaikin_smooth, args=(3, True))
        .apply(redistribute_vertices, args=(0.1,))
        # .translate(xoff=0, yoff=0)
    )

    lane_df_pl = (
        explode_linestring(lane_df)
        .assign(
            x=lambda df: df["points"].x,
            y=lambda df: df["points"].y,
        )
        .drop(columns=["points"])
        .pipe(lambda df: pl.from_pandas(df[["felt:id", "x", "y"]]))
        .with_columns(
            (pl.col("x") - pl.col("x").shift(1).backward_fill(limit=1))
            .over("felt:id")
            .alias("x_diff"),
            (pl.col("y") - pl.col("y").shift(1).backward_fill(limit=1))
            .over("felt:id")
            .alias("y_diff"),
        )
        .with_columns(
            # find the angle of the line segment
            pl.arctan2(pl.col("y_diff"), pl.col("x_diff")).alias("angle"),
        )
    )

    return lane_df, lane_df_pl


class Tree:
    def __init__(self, x, y, angles, dist_bound=3, max_angle_diff=10):
        self.tree = KDTree(np.column_stack([x, y]))
        self._angles = angles
        self._distance_bound = dist_bound
        self._max_angle_diff = max_angle_diff

    def query(
        self,
        pos_arr,
        angles,
    ):
        dist, inds = self.tree.query(
            pos_arr, p=2, distance_upper_bound=self._distance_bound
        )
        keep = dist != np.inf
        angle_diff = np.rad2deg(angles[keep] - self._angles[inds[keep]])
        angle_diff = (angle_diff + 180) % 360 - 180
        keep[keep] = np.abs(angle_diff) < self._max_angle_diff
        return dist[keep]


def build_tree(
    lane_df: pl.DataFrame, dist_bound: float = 6, max_angle_diff: float = 10
) -> Tree:
    return Tree(
        lane_df["x"].to_numpy(),
        lane_df["y"].to_numpy(),
        lane_df["angle"].to_numpy(),
        dist_bound=dist_bound,
        max_angle_diff=max_angle_diff,
    )


def rotate_vector(pos_arr, heading_arr, angle, offset):

    rot_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    ).squeeze()
    return (rot_matrix @ pos_arr.T).T + offset, heading_arr + angle


def solve_radar(
    ip: str,
    radar_df: pl.DataFrame,
    tree: Tree,
    radar_settings: dict,
    distance_range: tuple = (5, 200),
    sample_fraction: float = 0.5,
    angle_pm: float = 5,
    offset_pm: float = 3,
):

    single_radar_df = (
        radar_df.filter(pl.col("ip") == ip)
        .filter(pl.col("distance").is_between(*distance_range))
        .sample(fraction=sample_fraction)
    )

    radar_arr = single_radar_df.with_columns(
        pl.concat_list(
            [
                pl.col("f32_positionX_m"),
                pl.col("f32_positionY_m"),
            ],
        )
        .cast(pl.Array(inner=float, width=2))
        .alias("position_vect")
    )["position_vect"].to_numpy()

    origin = radar_settings[ip]["utm"]
    rot_angle = radar_settings[ip]["angle"]

    max_match = radar_arr.shape[0]

    loss_log = []

    def objective_func(sol: Tuple[float, float, float]):
        rot_angle, *origin_offset = sol

        d = tree.query(
            *rotate_vector(
                radar_arr,
                single_radar_df["heading"].to_numpy(),
                rot_angle,
                offset=np.array(origin) + np.array(origin_offset),
            )
        )

        match_loss = 1 - d.shape[0] / max_match  # the optimal here is 0
        average_d_loss = np.mean(d)  # the optimal here is 0

        # loss_log.append((sol, match_loss, average_d_loss))
        return match_loss + average_d_loss

    sol = minimize(
        objective_func,
        x0=(rot_angle, 0, 0),
        bounds=[
            (rot_angle - np.deg2rad(angle_pm), rot_angle + np.deg2rad(angle_pm)),
            (-1 * offset_pm, offset_pm),
            (-1 * offset_pm, offset_pm),
        ],
        tol=1e-9,
    )

    return sol, loss_log
