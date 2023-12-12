import numpy as np
import polars as pl
from src.filters.fusion import association_loglikelihood_distance
from src.pipelines.utils import lazify, timeit


@lazify
@timeit
def add_front_back_s(df: pl.DataFrame, use_median_length: bool = True) -> pl.DataFrame:
    required_columns = {"s", "distanceToFront_s", "distanceToBack_s", "length_s"}
    assert required_columns.issubset(
        df.columns
    ), f"Missing columns: {required_columns - set(df.columns)}"

    return df.with_columns(
        pl.col("length_s").median().over("object_id").alias("median_length_s")
    ).with_columns(
        (pl.col("s") + (pl.col("distanceToFront_s"))).alias("front_s"),
        # distance to back is negative
        (pl.col("s") + (pl.col("distanceToBack_s"))).alias("back_s"),
    )


@timeit
@lazify
def build_leader_follower_df(df: pl.DataFrame) -> pl.DataFrame:
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
    ]

    # create a lane_hash column
    df = df.with_columns(pl.struct(["lane", "lane_index"]).hash().alias("lane_hash"))

    # sort by epoch_time and s
    df = df.sort(
        ["epoch_time", "s"],
    ).set_sorted(["epoch_time", "s"])

    # shift over time to get the leader
    df = df.with_columns(
        pl.col("object_id").shift(-1).over(["epoch_time", "lane_hash"]).alias("leader")
    )

    # join the leader and follower
    return (
        df.select(["object_id", "epoch_time", "lane_hash", "leader", *keep_cols])
        .join(
            df.select(["object_id", "epoch_time", "lane_hash", *keep_cols]),
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
        .with_columns(
            (pl.col("s_leader") - pl.col("s")).alias("s_gap"),
        )
    )


@timeit
@lazify
def calc_assoc_liklihood_distance(
    df: pl.DataFrame,
    gpu: bool = True,
    batch_size: int = 100_000,
) -> pl.DataFrame:
    import torch

    dfs = []

    for _df in (
        df.with_row_count("maha_ind")
        .with_columns((pl.col("maha_ind") // batch_size).alias("chunk"))
        .partition_by("chunk")
    ):
        dfs.append(association_loglikelihood_distance(_df, gpu))

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
    alpha: float = 0.2,
) -> pl.DataFrame:
    return (
        df.lazy()
        .group_by(
            "pair",
        )
        .agg(
            pl.col("association_distance")
            .ewm_mean(alpha=alpha)
            .last()
            .alias("association_distance_filt"),
            pl.col("epoch_time").first().alias("epoch_time"),
            pl.col("prediction").any().alias("prediction"),
            pl.col("prediction_leader").any().alias("prediction_leader"),
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
            .then(pl.col("association_distance_filt") < assoc_cutoff)
            .otherwise(pl.col("association_distance_filt") < assoc_cutoff_pred)
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
    speed_threshold: float = 0.5,
    max_vehicle_num: int = 3,
) -> pl.DataFrame:
    ci_df = (
        df.lazy()
        .with_columns(
            pl.col("epoch_time").min().over("object_id").alias("start_time"),
        )
        .sort(["epoch_time", "start_time"])
        .with_columns(
            cumtime=pl.col("epoch_time").cum_count().over("object_id"),
            cumcount=pl.col("epoch_time")
            .cum_count()
            .over(["epoch_time", "vehicle_id"]),
            count=pl.col("epoch_time").count().over(["epoch_time", "vehicle_id"]),
        )
        .filter(
            ~((pl.col('count') > 1) &( pl.col('cumtime') < 10))
        )
        .drop(["cumtime", "cumcount", "count"])
        .with_columns(
            # cumtime=pl.col("epoch_time").cum_count().over("object_id"),
            cumcount=pl.col("epoch_time")
            .cum_count()
            .over(["epoch_time", "vehicle_id"]),
            count=pl.col("epoch_time").count().over(["epoch_time", "vehicle_id"]),
        )
        .filter(
            (
                # remove kalman filter errors in the radar
                ~(
                    (pl.col("count") > 1)
                    & pl.col("prediction")
                )
            )
        )
        # .with_columns(
        #     # count again after filtering
        #     cumcount=pl.col("epoch_time").count().over(["epoch_time", "vehicle_id"]),
        # )
        # .filter(
        #     (~pl.col("prediction") | (pl.col("cumcount") > 1))
        # )
        # .drop(["cumtime", "cumcount"])
        .collect()
    )

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
