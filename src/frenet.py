"""
Code for discretizing the Frenet Coordinate System

"""

from typing import Any, Tuple, TypeVar, Union
import numpy as np
from scipy.interpolate import splprep, splev
import geopandas as gpd
from scipy.spatial import KDTree
import polars as pl
from pyproj import CRS
from shapely.geometry import Point


T = TypeVar("T", bound="SplineLane")
D = TypeVar("D", bound="Detector")

CACHE = {}


class SplineLane:
    def __init__(
        self,
        name: str,
        centerline: np.ndarray,
        crs: str,
        width: float = 3.0,
    ) -> T:
        """
        Initialize a SplineLane object

        Args:
            name (str): the name of the lane
            centerline (np.ndarray): the centerline of the lane
            crs (str): the coordinate reference system of the lane
            width (float, optional): the width of the lane. Defaults to 3.0.

        """

        self.name: str = name
        self.crs: CRS = crs
        self._centerline: np.ndarray = self._prep_centerline(centerline)
        self.length: float = self._calc_length(centerline)
        self.width: float = width

        self._fit_params = {"s": None, "k": 3}

        self._fitted: bool = False

        self._tck: tuple = None
        self._u: np.ndarray = None

        self._discrete_s: np.ndarray = None
        self._ds_du: np.ndarray = None
        self._unew: np.ndarray = None

        self._kdtree: KDTree = None

        self._fitted_pl_df: pl.DataFrame = None

        self._detectors: list[D] = []

    @property
    def centerline(self) -> np.ndarray:
        """
        Get the centerline of the lane

        Returns:
            np.ndarray: the centerline of the lane
        """

        return self._centerline

    @property
    def s(self) -> np.ndarray:
        """
        Get the s coordinate of the lane

        Returns:
            np.ndarray: the s coordinate of the lane
        """

        return self._discrete_s * self.length

    @property
    def ds_du(self) -> np.ndarray:
        """
        Get the derivative of the centerline with respect to u

        Returns:
            np.ndarray: the derivative of the centerline with respect to u
        """

        return self._ds_du

    @property
    def u(self) -> np.ndarray:
        """
        Get the u coordinate of the lane

        Returns:
            np.ndarray: the u coordinate of the lane
        """

        return self._unew

    @property
    def kdtree(self) -> KDTree:
        if (self._kdtree is None) or (
            self._unew.shape[0] != self._kdtree.data.shape[0]
        ):
            self._kdtree = KDTree(self._unew, leafsize=20)
        return self._kdtree

    @property
    def detectors(self) -> list[D]:
        return self._detectors

    @staticmethod
    def _prep_centerline(centerline: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare the centerline for fitting. (Remove duplicate points)

        Args:
            centerline (np.ndarray): the centerline of the lane
        Returns:
            np.ndarray: the prepared centerline
        """

        # Remove duplicate points
        _, inds = np.unique(centerline, axis=0, return_index=True)
        # Sort the points
        xy = centerline[np.sort(inds), :]
        return (xy[:, 0], xy[:, 1])

    @staticmethod
    def _calc_length(centerline: np.ndarray) -> float:
        """
        Calculate the length of the centerline

        Args:
            centerline (np.ndarray): the centerline of the lane
        Returns:
            float: the length of the centerline
        """

        return np.sum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))

    def fit(self, k: int = 3, s: int = None, knots=None, task=0, *args, **kwargs) -> T:
        """
        Fit a spline to the centerline of the lane

        Args:
            s (float, optional): the smoothing factor. Defaults to 0.0.
            k (int, optional): the degree of the spline. Defaults to 3.
        Returns:
            T: the fitted SplineLane object

        """

        if self._centerline[0].shape[0] <= 3:
            k = 1

        if s is None:
            s = len(self._centerline[0]) - np.sqrt(2 * len(self._centerline[0]))

        self._tck, self._u = splprep(self._centerline, s=s, k=k, t=knots, task=task)
        self._fit_params["k"] = k
        self._fitted = True

        return self

    def interpolate(self, ds: float = 0.01) -> T:
        """
        Interpolate the centerline at a distance of ds

        Args:
            ds (float, optional): The interpolation size. Defaults to 0.01.

        Returns:
            T: the interpolated SplineLane object
        """

        # interpolate the centerline at a distance of ds
        num = int(self.length / ds)
        self._discrete_s = np.linspace(self._u.min(), self._u.max(), num)
        self._unew = np.array(splev(self._discrete_s, self._tck, der=0)).T
        self._ds_du = np.array(splev(self._discrete_s, self._tck, der=1)).T
        return self

    def to_gdf(
        self,
        linestring: bool = False,
        include_detectors: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Convert the lane to a geodataframe

        Returns:
            gpd.GeodataFrame: a geodataframe representation of the lane
        """
        from shapely.geometry import Point

        if linestring:
            from shapely.geometry import LineString

            multiplier = 1
            geometry = LineString(self._unew)
            other_cols = {}
        else:
            multiplier = len(self._unew)
            geometry = [Point(x, y) for x, y in self._unew]
            other_cols = {
                "x": self._unew[:, 0],
                "y": self._unew[:, 1],
                "s": self.s,
                "ds_du_x": self._ds_du[:, 0],
                "ds_du_y": self._ds_du[:, 1],
            }
        df = gpd.GeoDataFrame(
            {
                "name": [
                    self.name,
                ]
                * multiplier,
                "width": [
                    self.width,
                ]
                * multiplier,
                "geometry": geometry,
                **other_cols,
            },
            crs=self.crs,
        )

        if include_detectors:
            import pandas as pd

            df = pd.concat(
                [
                    df,
                    gpd.GeoDataFrame(
                        {
                            "name": [detector.name for detector in self._detectors],
                            "width": [0.0] * len(self._detectors),
                            "geometry": [
                                Point(detector.pos) for detector in self._detectors
                            ],
                            "detector": [
                                detector.detector for detector in self._detectors
                            ],
                        },
                        crs=self.crs,
                    ),
                ]
            )

        return df

    # def get_directed_d(self, pos: Tuple[Union[float, np.ndarray], ...], d: Union[float, np.ndarray], inds: Union[float, np.ndarray]) -> float:

    #     """
    #     Get the directed distance between the position and the lane

    #     Args:
    #         pos (Tuple[Any[float, np.ndarray], ...]): the position
    #         d (Any[float, np.ndarray]): the distance between the position and the lane
    #         inds (Any[float, np.ndarray]): the indices of the closest points on the lane
    #     Returns:
    #         float: the directed distance between the position and the lane
    #     """
    #     u = self._ds_du[inds]
    #     v = pos - self._unew[inds]

    #     # get the angle between the point and the centerline, relative to the derivative
    #     # of the centerline
    #     angle = np.arccos(
    #         np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v)),
    #     )
    #     # if it is a negative angle,
    #     # then the point is on the right side of the centerline
    #     # and we need to flip the sign of the distance
    #     d = np.where(angle < np.pi / 2, d, -d)

    #     return d

    def snap_points(
        self,
        points: np.ndarray,
        max_dist: float = 6,
        return_all: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Snap Radar Data Points to the Lane

        Args:
            points (np.ndarray): the radar data points (x, y)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                the closest points on the lane (x, y),
                the s coordinate of the closest points,
                the derivative of the centerline with respect to u,
                the distance between the closest points and the radar data points
        """
        d, idx = self.kdtree.query(
            points,
            p=2,
            workers=-1,
            distance_upper_bound=max_dist,
        )  # p=2 for euclidean distance

        slicer = idx < self.kdtree.n
        true_idx = idx[slicer]
        angle = np.empty_like(d)

        u = self._ds_du[true_idx]
        v = points[true_idx] - self._unew[true_idx]

        # get the angle between the point and the centerline, relative to the derivative
        # of the centerline
        angle[slicer] = np.arccos(
            np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v)),
        )
        # if it is a negative angle,
        # then the point is on the right side of the centerline
        # and we need to flip the sign of the distance
        d[slicer] = np.where(angle[slicer] < np.pi / 2, d[slicer], -d[slicer])

        if return_all:
            # get the distance between the point and the centerline
            return self._unew[idx], self.s[idx], self._ds_du[idx], d

        return idx, d

    @property
    def fitted_pl_df(self) -> pl.DataFrame:
        if self._fitted_pl_df is None:
            self._fitted_pl_df = pl.DataFrame(
                {
                    "x": self._unew[:, 0],
                    "y": self._unew[:, 1],
                    "s": self.s,
                    "ds_du_x": self._ds_du[:, 0],
                    "ds_du_y": self._ds_du[:, 1],
                }
            )

        return self._fitted_pl_df

    def get_s(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        d, ind = self.kdtree.query(pos, p=2, workers=-1)
        return self.s[ind], d

    def add_detector(
        self, pos: Tuple[float, float], crs: CRS, attrs: dict = None
    ) -> None:
        if attrs is None:
            attrs = {}

        self._detectors.append(
            Detector(
                pos=pos,
                lane=self,
                crs=crs,
                attrs=attrs,
            )
        )

    def add_detectors_from_file(self, path_2_detectors: str) -> T:
        import geopandas as gpd

        if str(path_2_detectors) in CACHE:
            detectors = CACHE[str(path_2_detectors)]
        else:
            detectors = gpd.read_file(path_2_detectors, ARRAY_AS_STRING="YES").to_crs(
                self.crs
            )
            CACHE[str(path_2_detectors)] = detectors

        l_s = self.to_gdf(linestring=True)["geometry"].iloc[0]
        # convert the lane to a geodataframe
        detectors = detectors.loc[detectors.intersects(l_s)].copy()

        detectors["i"] = detectors.intersection(l_s)

        detectors = detectors.dropna(subset=["i"])

        for _, detector in detectors.iterrows():
            if detector["i"].geom_type == "MultiPoint":
                detector["i"] = detector["i"]

            self.add_detector(
                pos=(detector["i"].x, detector["i"].y),
                # name=detector.id,
                crs=self.crs,
                attrs=detector.drop(["geometry", "i"]).to_dict(),
            )

        return self

    def check_crosses(
        self,
        df: pl.DataFrame,
        frenet_s_col: str,
        car_cols: Tuple[str],
        lane_col: str = "lane",
    ) -> pl.DataFrame:
        df = (
            df.lazy()
            # .sort(
            #     by=[pl.col("epoch_time"), *car_cols],
            # )
            # .set_sorted(
            #     [pl.col("epoch_time"), *car_cols],
            # )
            # .with_row_count()
        )

        if "next_s" not in df.columns:
            df = df.with_columns([pl.col(frenet_s_col).shift(-1).alias("next_s")])

        df = (
            df
            # .filter(pl.col(lane_col) == self.name)
            .with_columns(
                [
                    detector.check_cross_expr(
                        df,
                        frenet_s_col=frenet_s_col,
                        car_cols=car_cols,
                        lane_col=lane_col,
                    )
                    for detector in self._detectors
                ]
            )
        )

        # # if lane_col in check_df.columns:
        # #     check_df = check_df.drop(["next_s"])
        return df.collect()
        # return df.join(
        #     check_df.select(["row_nr", *(detector.name for detector in self._detectors)]),
        #     on="row_nr",
        #     how="left",
        # ).collect()

    def check_crosses_grouped_expr(
        self,
        lane_col: str = "lane",
        return_dist: bool = False,
        return_index: bool = False,
    ) -> Tuple[pl.Expr, ...]:
        return (
            expr
            for detector in self._detectors
            for expr in detector.check_cross_expr_grouped(
                lane_col=lane_col,
                return_dist=return_dist,
                return_index=return_index,
            )
            if expr is not None
        )

    def build_detector_df(
        self,
    ) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "name": detector.name,
                    "pos": detector.pos,
                    "subtracts": detector.subtract_detectors,
                    "lane": self.name,
                }
                for detector in self._detectors
            ]
        )


class Detector:
    def __init__(
        self,
        # name: str,
        pos: Tuple[float, float],
        lane: SplineLane,
        crs: CRS,
        attrs: dict,
    ) -> None:
        """
        Initialize a Detector object.

        Args:
            pos (Tuple[float, float]): the lat/lon position of the detector (
                this should fall on the lane)
            lane (SplineLane): _description_
        """

        self.pos = pos
        self._lane = lane

        self._utm_pos = self._convert_2_utm(crs=crs)
        self._s = self._calc_s()

        self.attrs = attrs
        self.radar = self.attrs.get("radar", None)

        if not self.radar:
            raise ValueError("The detector must have a radar")

        if isinstance(self.radar, (float, int)):
            try:
                self.radar = str(int(self.radar))
            except ValueError:
                print(attrs)

        self.detector = self.attrs.get("detector", None)
        if isinstance(self.detector, dict):
            self.detector = self.detector.get(self._lane.name, None)
        if not self.detector:
            raise ValueError("The detector must have a detector")

        self.tl = self.attrs.get("tl", None)
        # handle the fuuuucked up pd nan types
        if isinstance(self.tl, float) and np.isnan(self.tl):
            self.tl = None
        elif isinstance(self.tl, str) and self.tl.lower() == "nan":
            self.tl = None
        else:
            self.tl = int(self.tl)

        # if not self.tl:
        #     raise ValueError("The detector must have a tl")

        if isinstance(self.tl, (float, str, int)) and isinstance(
            self.detector, (float, str, int)
        ):
            self.name = f"{self.tl}-{self.detector}"
        else:
            self.name = self.detector

        # get the exclusive tag
        self.subtract_detectors = self.attrs.get("subtract_detectors", [])

        if isinstance(self.subtract_detectors, str):
            self.subtract_detectors = list(
                map(
                    lambda x: int(x.strip()),
                    self.subtract_detectors.replace("[", "")
                    .replace("]", "")
                    .split(","),
                )
            )
        else:
            self.subtract_detectors = [0]

    def __repr__(self) -> str:
        return f"Detector(name={self.name}, pos={self.pos}, s={self._s}, lane={self._lane.name}, attrs={self.attrs})"

    @property
    def s(self) -> float:
        return self._s

    def _convert_2_utm(self, crs: CRS) -> None:
        """
        Convert the lat/lon position to utm
        """
        import utm

        # if the crs is already utm, then just return the position
        if crs.is_projected:
            return self.pos

        try:
            x, y, _, _ = utm.from_latlon(
                self.pos[0],
                self.pos[1],
            )
        except utm.OutOfRangeError:
            x, y, _, _ = utm.from_latlon(
                self.pos[1],
                self.pos[0],
            )

        return x, y

    def _calc_s(self) -> None:
        """
        Calculate the s coordinate of the detector
        """
        s, d = self._lane.get_s(self._utm_pos)

        return s

    def check_cross_expr_grouped(
        self,
        frenet_s_col: str = "s",
        lane_col: str = "lane",
        return_dist: bool = False,
        return_index: bool = False,
    ) -> pl.DataFrame:
        return [
            (
                pl.when(
                    (pl.col(lane_col).first() == self._lane.name)
                    & pl.col("ip").first().str.contains(self.radar)
                )
                .then(
                    pl.col("row_nr").take(
                        (pl.col(frenet_s_col) - pl.lit(self._s)).abs().arg_min()
                    )
                )
                .otherwise(pl.lit(None))
                .alias(f"ind-{self.name}")
            )
            if return_index
            else None,
            (
                pl.when(
                    (pl.col(lane_col).first() == self._lane.name)
                    & pl.col("ip").first().str.contains(self.radar)
                )
                .then((pl.col(frenet_s_col) - pl.lit(self._s)).abs().min())
                .otherwise(pl.lit(None))
                .alias(f"mins-{self.name}")
            )
            if return_dist
            else None,
        ]
