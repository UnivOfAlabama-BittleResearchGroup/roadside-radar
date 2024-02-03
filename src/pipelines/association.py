from datetime import timedelta
import numpy as np
import polars as pl
from src.filters.fusion import association_loglikelihood_distance
from src.pipelines.utils import lazify, timeit


@lazify
@timeit
def add_front_back_s(
    df: pl.DataFrame,
    use_median_length: bool = False,
    use_nearest_length: bool = False,
    s_col: str = "s",
) -> pl.DataFrame:
    required_columns = {s_col, "distanceToFront_s", "distanceToBack_s", "length_s"}
    assert required_columns.issubset(
        df.columns
    ), f"Missing columns: {required_columns - set(df.columns)}"

    if use_median_length:
        # multiply the median length by the percent front and back
        raise NotImplementedError
        # return (
        #     df.with_columns(
        #         pl.col("length_s").median().over("object_id").alias("median_length_s"),
        #         (pl.col("distanceToFront_s") / pl.col("length_s")).alias(
        #             "front_percent"
        #         ),
        #         (pl.col("distanceToBack_s") / pl.col("length_s")).alias("back_percent"),
        #     )
        #     .with_columns(
        #         (pl.col("median_length_s") * pl.col("front_percent")).alias("front_s"),
        #         (pl.col("median_length_s") * pl.col("back_percent")).alias("back_s"),
        #     )
        #     .drop(["median_length_s", "front_percent", "back_percent"])
        # )

    if use_nearest_length:
        # if approaching, we can assume that the distanceToFront_s is right,
        # and the back is found by subtracting the length
        # the length is found by taking the reading that is closest to the radar.
        assert "dist" in df.columns, 'Must have "dist" column'
        assert "approaching" in df.columns, 'Must have "approaching" column'
        return (
            df.with_columns(
                pl.col("length_s")
                .sort_by(pl.col("dist"))
                .first()
                .over("object_id")
                .alias("nearest_length_s"),
            )
            .with_columns(
                pl.when(pl.col("approaching"))
                .then(pl.col("distanceToFront_s") + pl.col(s_col))
                .otherwise(
                    (
                        pl.col("distanceToBack_s")
                        + pl.col(s_col)
                        + pl.col("nearest_length_s")
                    )
                )
                .alias("front_s"),
                pl.when(pl.col("approaching"))
                .then(
                    (
                        pl.col("distanceToFront_s")
                        + pl.col(s_col)
                        - pl.col("nearest_length_s")
                    )
                )
                .otherwise(pl.col("distanceToBack_s") + pl.col(s_col))
                .alias("back_s"),
            )
            .drop("nearest_length_s")
        )

    if "front" in s_col:
        return df.with_columns(
            # centroid
            (pl.col(s_col) - pl.col("distanceToFront_s")).alias("s"),
            # back
            (
                pl.col(s_col)
                - pl.max_horizontal(
                    pl.col("length_s"),
                    pl.col("distanceToFront_s") + -1 * pl.col("distanceToBack_s"),
                )
            ).alias("back_s"),
        )
    elif s_col == "s":
        return df.with_columns(
            (pl.col(s_col) + (pl.col("distanceToFront_s")))
            # .clip_min(pl.col(s_col))
            .alias("front_s"),
            # distance to back is negative
            (pl.col(s_col) + (pl.col("distanceToBack_s")))
            # .clip_max(pl.col(s_col))
            .alias("back_s"),
        )
    else:
        raise NotImplementedError


@timeit
@lazify
def build_leader_follower_df(df: pl.DataFrame, s_col: str = "s") -> pl.DataFrame:
    if s_col not in {"s", "front_s", "back_s"}:
        raise ValueError("s_col must be one of {'s', 'front_s', 'back_s'}")

    keep_cols = [
        "s",
        "front_s",
        "back_s",
        "s_velocity",
        "d",
        "d_velocity",
        "P",
        "prediction",
        "ip",
        "length_s",
        "lane_index",
    ]

    # create a lane_hash column
    df = df.with_columns(pl.struct(["lane", "lane_index"]).hash().alias("lane_hash"))

    # sort by epoch_time and s
    df = df.sort(
        ["epoch_time", "lane_hash", s_col],
    ).set_sorted(["epoch_time", "lane_hash", s_col])

    # shift over time to get the leader
    df = df.with_columns(
        pl.col("object_id").shift(-1).over(["epoch_time", "lane_hash"]).alias("leader")
    )

    # join the leader and follower
    return (
        df.select(["object_id", "epoch_time", "lane_hash", "leader", *keep_cols])
        .join(
            df.select(["object_id", "epoch_time", "lane_hash", *keep_cols]).drop(
                "lane_index"
            ),
            left_on=[
                "leader",
                "epoch_time",
            ],
            right_on=[
                "object_id",
                "epoch_time",
            ],
            how="inner",
            suffix="_leader",
        )
        .sort("epoch_time")
        .set_sorted("epoch_time")
        .with_columns(
            (pl.col(f"{s_col}_leader") - pl.col(s_col)).alias("s_gap"),
        )
    )


@timeit
@lazify
def calc_assoc_liklihood_distance(
    df: pl.DataFrame,
    gpu: bool = True,
    batch_size: int = 100_000,
    dims: int = 4,
) -> pl.DataFrame:
    import torch

    dfs = []

    for _df in (
        df.with_row_count("maha_ind")
        .with_columns((pl.col("maha_ind") // batch_size).alias("chunk"))
        .partition_by("chunk")
    ):
        dfs.append(association_loglikelihood_distance(_df, gpu, dims))

    try:
        torch.cuda.empty_cache()
    except:  # noqa: E722
        pass

    return pl.concat(dfs).drop(["maha_ind", "chunk"])


@timeit
@lazify
def calculate_match_indexes(
    df: pl.DataFrame,
    match_time_threshold: float = 1,
) -> pl.DataFrame:
    return (
        df.lazy()
        .with_columns(
            # sort the object id and leader
            pl.struct(
                [
                    pl.col("leader"),
                    pl.col("object_id"),
                ]
            )
            # .hash()
            .alias("pair")
        )
        .with_columns(
            (pl.col("prediction") | pl.col("prediction_leader")).alias(
                "prediction_any"
            ),
            (
                (pl.col("epoch_time") - pl.col("epoch_time").min()).dt.milliseconds()
                / 1e3
            )
            .over("pair")
            .alias("match_time"),
        )
        .filter(pl.col("match_time") < match_time_threshold)
        .sort(["pair", "match_time"])
        .with_columns(
            [
                # ----------- Calculate the Time Headway ------------
                ((pl.col("s_leader") - pl.col("s")) / pl.col("s_velocity")).alias(
                    "headway"
                ),
                # -------- Calculate the Match Indexes Take 2 ------------
                # Find periods where there is no prediction
                # prediction naturally has more uncertainty
                # ---------------------------
                # find the start index to take
                pl.col("object_id").cumcount().over("pair").alias("sort_index"),
                # find the end index to use
            ]
        )
        .with_columns(
            pl.when(pl.col("prediction_any"))
            .then(pl.col("sort_index").max())
            .otherwise(pl.col("sort_index"))
            .over("pair")
            .alias("sort_index"),
        )
        .sort(["pair", "sort_index", "epoch_time"], descending=[False, False, True])
    )


@timeit
@lazify
def pipe_gate_headway_calc(
    df: pl.DataFrame,
    alpha: float = 0.9,
) -> pl.DataFrame:
    return (
        df.lazy()
        .group_by(
            "pair",
        )
        .agg(
            pl.col("association_distance")
            .ewm_mean(alpha=alpha)
            .min()
            .alias("association_distance_filt"),
            pl.col("epoch_time").first().alias("epoch_time"),
            pl.col("prediction").any().alias("prediction"),
            pl.col("prediction_leader").any().alias("prediction_leader"),
            pl.col("headway").min().alias("headway"),
        )
    )


@timeit
@lazify
def build_match_df(
    df: pl.DataFrame,
    traj_time_df: pl.DataFrame,
    assoc_cutoff: float,
    assoc_cutoff_pred: float,
) -> pl.DataFrame:
    valid_matches = (
        df.lazy()
        .sort("epoch_time")
        .unnest("pair")
        .with_row_count()
        .join(
            traj_time_df.lazy(),
            on="object_id",
        )
        .join(
            traj_time_df.lazy(),
            left_on="leader",
            right_on="object_id",
            suffix="_leader",
        )
        .collect()
    )

    keep_rows = (
        valid_matches.lazy()
        .melt(
            id_vars=[
                "epoch_time",
                "row_nr",
                "prediction",
                "prediction_leader",
                "epoch_time_max",
                "epoch_time_max_leader",
                "association_distance_filt",
            ]
        )
        .filter(
            pl.when(~(pl.col("prediction") | pl.col("prediction_leader")))
            .then(pl.col("association_distance_filt") <= assoc_cutoff)
            .otherwise(pl.col("association_distance_filt") <= assoc_cutoff_pred)
        )
        .sort("value", "epoch_time")
        .with_columns(
            pl.when(pl.col("variable") == "object_id")
            .then(pl.col("prediction"))
            .otherwise(pl.col("prediction_leader"))
            .alias("prediction"),
            pl.when(pl.col("variable") == "object_id")
            .then(pl.col("epoch_time_max"))
            .otherwise(pl.col("epoch_time_max_leader"))
            .alias("my_end_time"),
            pl.when(pl.col("variable") == "object_id")
            .then(pl.col("epoch_time_max_leader"))
            .otherwise(pl.col("epoch_time_max"))
            .alias("other_end_time"),
        )
        .drop(
            ["epoch_time_max", "epoch_time_max_leader", "prediction_leader", "variable"]
        )
        .with_columns(
            pl.col("prediction").cumsum().over("value").alias("prediction_count"),
            pl.col("other_end_time")
            .filter(~pl.col("prediction"))
            .max()
            .over("value")
            .alias("other_end_time_max"),
        )
        .filter(
            (pl.col("prediction_count") <= 1)
            & (pl.col("row_nr").count().over("row_nr") > 1)
        )
        .collect()
    )

    return valid_matches.filter(
        pl.col("row_nr").is_in(keep_rows["row_nr"].unique())
    ).drop("row_nr")


@timeit
@lazify
def create_vehicle_ids(
    df: pl.DataFrame,
    match_df: pl.DataFrame,
) -> pl.DataFrame:
    import networkx as nx

    # create a bidirectional graph of connections, with the weight being the time difference
    g = nx.Graph(
        [
            (d["object_id"], d["leader"])
            for d in match_df.select(
                [
                    "object_id",
                    "leader",
                ]
            ).to_dicts()
        ]
    )

    # get all the connected components
    cc = list(nx.connected_components(g))

    # create a dataframe of the connected components
    cc_df = pl.DataFrame(
        {
            "object_id": [int(_x) for x in cc for _x in x],
            "vehicle_id": [i for i, x in enumerate(cc) for _ in x],
        },
        schema={
            "object_id": pl.UInt64,
            "vehicle_id": pl.UInt64,
        },
    )

    cc_df = cc_df.vstack(
        df.select(pl.col("object_id").unique())
        .filter(~pl.col("object_id").is_in(cc_df["object_id"]))
        .with_row_count(
            "vehicle_id",
            offset=cc_df["vehicle_id"][-1] + 1,
        )
        .cast({"vehicle_id": pl.UInt64})
        .select(["object_id", "vehicle_id"])
    )

    return df.join(cc_df, on="object_id", how="left")


@timeit
@lazify
def build_fusion_df(
    df: pl.DataFrame,
    max_vehicle_num: int = 3,
    prediction_length: float = 4,
) -> pl.DataFrame:
    ci_df = (
        df.lazy()
        # label the start time of each object
        .with_columns(
            pl.col("epoch_time").min().over("object_id").alias("start_time"),
        )
        # sort by the start time
        .sort(["epoch_time", "start_time"])
        .with_columns(
            # count the number of times the object has been seen
            cumtime=pl.col("epoch_time").cum_count().over("object_id"),
            # count the number of measurements per vehicle-time
            cumcount=pl.col("epoch_time")
            .cum_count()
            .over(["epoch_time", "vehicle_id"]),
            # count the total number of measurements per vehicle
            count=pl.col("epoch_time").count().over(["epoch_time", "vehicle_id"]),
            timedelta=(pl.col("epoch_time").max() - pl.col("epoch_time"))
            .dt.total_milliseconds()
            .over("vehicle_id"),
        )
        # filter out the first second of measurements if the count > 0
        # .filter((pl.col('cumtime') > 10) | (pl.col('cumcount') < 1))
        # # .drop(["cumtime", "cumcount", "count"])
        .with_columns(
            # cumtime=pl.col("epoch_time").cum_count().over("object_id"),
            cumcount=pl.col("epoch_time")
            .cum_count()
            .over(["epoch_time", "vehicle_id"]),
            count=pl.col("epoch_time").count().over(["epoch_time", "vehicle_id"]),
        )
        # .filter(
        #     (
        #         # remove kalman filter errors in the radar
        #         ~((pl.col("cumcount") == 0) & pl.col("prediction") & (pl.col("count") > 1))
        #     )
        # )
        # .filter(
        #     (
        #         ~((pl.col('count') > 1) & pl.col('prediction'))
        #     )
        # )
        # .with_columns(
        #     # count again after filtering
        #     # cumcount=pl.col("epoch_time").count().over(["epoch_time", "vehicle_id"]),
        # )
        # .filter(
        #     (~pl.col("prediction") | (pl.col("cumcount") > 1))
        # )
        .filter(
            ~(
                (pl.col("count") == 1)
                & pl.col("prediction")
                & (pl.col("timedelta") < prediction_length * 1e3)
            )
        )
        .drop(["cumtime", "cumcount"])
        .collect()
    )

    # ci_df = ci_df.filter(
    #     (
    #         (pl.col("epoch_time").max() - pl.col("epoch_time")) > timedelta(seconds=4)
    #     ).over("vehicle_id")
    # )

    ci_df = (
        ci_df.lazy()
        .sort(["vehicle_id", "epoch_time"])
        # .set_sorted(["vehicle_id", "epoch_time"])
        .join(
            ci_df.select(["vehicle_id", "epoch_time"])
            .lazy()
            .unique()
            .sort(["vehicle_id", "epoch_time"])
            .with_columns(
                (pl.col("epoch_time").cumcount()).over("vehicle_id").alias("time_index")
            ),
            on=["vehicle_id", "epoch_time"],
        )
        .with_columns(
            pl.col("epoch_time").first().over("vehicle_id").alias("vehicle_start_time"),
        )
        # sort by the start times
        .sort(
            [
                "prediction",
                "vehicle_start_time",
            ]
        )
        # .set_sorted(["vehicle_id", "vehicle_start_time", "epoch_time"])
        .with_columns(
            pl.col("object_id")
            .cumcount()
            .over(["vehicle_id", "time_index"])
            .alias("vehicle_time_index_int")
        )
        .filter(pl.col("vehicle_time_index_int") < max_vehicle_num)
        .sort("epoch_time")
        # .set_sorted("epoch_time")
        .with_columns(
            (pl.col("epoch_time").diff().dt.total_milliseconds() / 1000)
            .cast(float)
            .over(
                "object_id",
            )
            .fill_null(0)
            .alias("time_diff")
        )
        .drop(["vehicle_start_time", "time_ind", "vehicle_ind"])
    )

    return ci_df


@lazify
@timeit
def filter_bad_lane_matches(
    df: pl.DataFrame, traj_df: pl.DataFrame, s_threshold: float = 20
) -> pl.DataFrame:
    return df.filter(
        pl.col("pair").is_in(
            df.lazy()
            .drop("epoch_time")
            .unnest("pair")
            .join(
                traj_df.select(
                    [
                        "object_id",
                        "epoch_time",
                        "lane",
                        "lane_index",
                        "s",
                    ]
                ).lazy(),
                how="left",
                left_on="leader",
                right_on="object_id",
            )
            .join(
                traj_df.select(
                    [
                        "object_id",
                        "epoch_time",
                        "lane",
                        "lane_index",
                        "s",
                    ]
                ).lazy(),
                how="left",
                left_on=["object_id", "epoch_time"],
                right_on=["object_id", "epoch_time"],
            )
            .group_by(["object_id", "leader"])
            .agg(
                (
                    # (pl.col("lane") == pl.col("lane_right"))
                    (pl.col("lane_index") == pl.col("lane_index_right"))
                    | pl.col("lane_index").is_null()
                    | pl.col("lane_index_right").is_null()
                )
                .all()
                .alias("lane_match"),
                (
                    ((pl.col("s") - pl.col("s_right")).abs() < s_threshold)
                    | pl.col("s").is_null()
                    | pl.col("s_right").is_null()
                )
                .all()
                .alias("s_match"),
            )
            .filter(pl.col("lane_match") & pl.col("s_match"))
            .with_columns(
                pl.struct(["leader", "object_id"]).alias("pair"),
            )
            .collect(streaming=True)["pair"]
        )
    )
