import polars as pl
from src.filters.metrics import association_loglikelihood_distance
from src.pipelines.utils import lazify, timeit
from scipy.stats import chi2
import networkx as nx


@lazify
@timeit
def add_front_back_s(
    df: pl.DataFrame,
    use_median_length: bool = False,
    use_nearest_length: bool = False,
    use_global_mean: bool = False,
    s_col: str = "s",
) -> pl.DataFrame:
    required_columns = {s_col, "distanceToFront_s", "distanceToBack_s", "length_s"}
    assert required_columns.issubset(
        df.columns
    ), f"Missing columns: {required_columns - set(df.columns)}"

    if use_median_length:
        # multiply the median length by the percent front and back
        raise NotImplementedError

    if use_global_mean:
        mean_length = df.select(
            pl.col("length_s").filter(pl.col("dist").is_between(10, 50)).mean()
        ).collect()["length_s"][0]

        return (
            df.with_columns(
                pl.when((pl.col("dist") >= 50) & (pl.col("length_s") < mean_length))
                .then(pl.lit(mean_length))
                .otherwise(pl.col("length_s"))
                .alias("corrected_length_s")
            )
            .with_columns(
                pl.when(pl.col("approaching"))
                .then(pl.col("distanceToFront_s") + pl.col("s"))
                .otherwise(
                    (
                        pl.col("distanceToBack_s")
                        + pl.col("s")
                        + pl.col("corrected_length_s")
                    )
                )
                .alias("front_s"),
                pl.when(pl.col("approaching"))
                .then(
                    (
                        pl.col("distanceToFront_s")
                        + pl.col("s")
                        - pl.col("corrected_length_s")
                    )
                )
                .otherwise(pl.col("distanceToBack_s") + pl.col("s"))
                .alias("back_s"),
            )
            .drop("corrected_length_s")
        )

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
def build_leader_follower_df(
    df: pl.DataFrame, s_col: str = "s", use_lane_index: bool = True
) -> pl.DataFrame:
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
    df = df.with_columns(
        pl.struct(["lane", "lane_index"]).hash().alias("lane_hash")
        if use_lane_index
        else pl.struct(["lane"]).hash().alias("lane_hash")
    )

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
def build_leader_follower_entire_history_df(
    df: pl.DataFrame,
    s_col: str = "s",
    use_lane_index: bool = True,
    max_s_gap: float = 20,
) -> pl.DataFrame:
    if s_col not in {"s", "front_s", "back_s"}:
        raise ValueError("s_col must be one of {'s', 'front_s', 'back_s'}")

    keep_cols = [
        "s",
        "front_s",
        "back_s",
        "s_velocity",
        "s_accel",
        "d",
        "d_velocity",
        "d_accel",
        "P",
        "prediction",
        "ip",
        "length_s",
        "lane_index",
    ]

    # create a lane_hash column
    lf_df = (
        df.with_columns(
            pl.struct(["lane", *(("lane_index",) if use_lane_index else ())])
            .hash()
            .alias("lane_hash")
        )
        .sort(
            ["epoch_time", "lane_hash", s_col],
        )
        .set_sorted(["epoch_time", "lane_hash", s_col])
        .with_columns(
            pl.col(["object_id", s_col])
            .shift(-1)
            .over(["epoch_time", "lane_hash"])
            .name.map(lambda x: x + "_leader")
        )
        .filter((pl.col(f"{s_col}_leader") - pl.col(s_col)) < max_s_gap)
        .select(
            [
                "object_id",
                "lane",
                pl.col("object_id_leader").alias("leader"),
            ]
        )
        .unique()
    )

    # join all the data
    return (
        lf_df.join(
            df.select(["object_id", "epoch_time", "lane", *keep_cols]),
            on=[
                "object_id",
                "lane",
            ],
        )
        .join(
            df.select(["object_id", "epoch_time", "lane", *keep_cols]),
            left_on=[
                "leader",
                "epoch_time",
                "lane",
            ],
            right_on=[
                "object_id",
                "epoch_time",
                "lane",
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


@lazify
def build_leader_follower_no_sort(
    df: pl.DataFrame,
    s_col: str = "s",
    max_s_gap=200,
    use_lane_index: bool = True,
) -> pl.DataFrame:
    keep_cols = [
        "s",
        "front_s",
        "back_s",
        "s_velocity",
        "s_accel",
        "d",
        "d_velocity",
        "d_accel",
        "P",
        "prediction",
        "ip",
        "length_s",
        "lane_index",
    ]

    df = df.with_columns(
        pl.struct(["lane", *(("lane_index",) if use_lane_index else ())])
        .hash()
        .alias("lane_hash")
    ).collect()

    unique_pairs = []
    # partion by the hour
    for _df in df.with_columns(
        pl.col("epoch_time").dt.date().alias("date"),
        pl.col("epoch_time").dt.hour().alias("hour"),
    ).partition_by(["date", "hour"]):
        unique_pairs.append(
            _df.select(["epoch_time", "lane_hash", s_col, "object_id"])
            .lazy()
            .join(
                _df.select(["epoch_time", "lane_hash", s_col, "object_id"]).lazy(),
                on=["epoch_time", "lane_hash"],
                suffix="_other",
            )
            .filter(pl.col("object_id") != pl.col("object_id_other"))
            .filter((pl.col(f"{s_col}_other") - pl.col(s_col)) < max_s_gap)
            .select(
                [
                    "object_id",
                    pl.col("object_id_other").alias("leader"),
                ]
            )
            .with_columns(
                pl.when(pl.col("object_id") < pl.col("leader"))
                .then(pl.concat_list([pl.col("object_id"), pl.col("leader")]))
                .otherwise(pl.concat_list([pl.col("leader"), pl.col("object_id")]))
                .alias("pair"),
            )
            .select(
                [
                    "pair",
                ]
            )
            .with_columns(
                pl.col("pair").list.get(0).alias("object_id"),
                pl.col("pair").list.get(1).alias("leader"),
            )
            .drop("pair")
            .unique()
            .join(
                _df.lazy().select(["object_id", "epoch_time", "lane", *keep_cols]),
                on="object_id",
            )
            .join(
                _df.lazy().select(["object_id", "epoch_time", "lane", *keep_cols]),
                left_on=[
                    "leader",
                    "epoch_time",
                    "lane",
                ],
                right_on=[
                    "object_id",
                    "epoch_time",
                    "lane",
                ],
                how="inner",
                suffix="_leader",
            )
            .sort("epoch_time")
            .set_sorted("epoch_time")
            .with_columns(
                (pl.col(f"{s_col}_leader") - pl.col(s_col)).alias("s_gap"),
            )
            # .collect(streaming=True)
        )

    exploded_df = pl.concat(unique_pairs).unique(
        ["object_id", "leader", 'epoch_time']
    )

    return exploded_df


@timeit
@lazify
def calc_assoc_liklihood_distance(
    df: pl.DataFrame,
    gpu: bool = True,
    batch_size: int = 100_000,
    dims: int = 4,
    permute: bool = False,
    maha: bool = False,
) -> pl.DataFrame:
    import torch

    dfs = []

    for _df in (
        df.with_row_count("maha_ind")
        .with_columns((pl.col("maha_ind") // batch_size).alias("chunk"))
        .partition_by("chunk")
    ):
        dfs.append(association_loglikelihood_distance(_df, gpu, dims, permute, maha=maha))

    try:
        torch.cuda.empty_cache()
    except:  # noqa: E722
        pass

    return pl.concat(dfs).drop(["maha_ind", "chunk"])


@timeit
@lazify
def calculate_match_indexes(
    df: pl.DataFrame,
) -> pl.DataFrame:
    return df.lazy().with_columns(
        pl.when(pl.col("leader") < pl.col("object_id"))
        .then(pl.concat_list([pl.col("leader"), pl.col("object_id")]))
        .otherwise(pl.concat_list([pl.col("object_id"), pl.col("leader")]))
        .alias("pair")
    )


@timeit
@lazify
def pipe_gate_headway_calc(
    df: pl.DataFrame,
    window: int = 20,
    association_dist_cutoff: float = chi2.ppf(0.95, 4),
) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("association_distance")
            .rolling_mean(window_size=window, min_periods=1)
            .over("pair")
        )
        .group_by("pair")
        .agg(
            # find the minimum assoc. distance
            pl.col("association_distance").min().alias("association_distance_filt"),
            # find the time that they are first joined
            pl.col("epoch_time")
            .filter(pl.col("association_distance") < association_dist_cutoff)
            .first()
            .alias("join_time"),
            # identify the first time that are in following
            pl.col("epoch_time").first().alias("epoch_time"),
            # identify the max time in following
            pl.col("epoch_time").last().alias("epoch_time_max"),
            #
            pl.col("prediction")
            .filter(pl.col("association_distance") < association_dist_cutoff)
            .first()
            .alias("prediction"),
            pl.col("prediction_leader")
            .filter(pl.col("association_distance") < association_dist_cutoff)
            .first()
            .alias("prediction_leader"),
            pl.col("s_leader").first().alias("leader_s"),
            pl.col("s").first().alias("s"),
        )
    )


@timeit
@lazify
def build_match_df(
    df: pl.DataFrame,
    traj_time_df: pl.DataFrame,
    assoc_cutoff: float,
    assoc_cutoff_pred: float,
    # headway_cutoff_inner: float,
) -> pl.DataFrame:
    return (
        df
        # .lazy()
        .sort("epoch_time")
        # .unnest("pair")
        .with_columns(
            pl.col("pair").list.get(0).alias("object_id"),
            pl.col("pair").list.get(1).alias("leader"),
        )
        # .with_row_count()
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
        .filter(
            (
                pl.when(~(pl.col("prediction") | pl.col("prediction_leader")))
                .then(pl.col("association_distance_filt") <= assoc_cutoff)
                .otherwise(pl.col("association_distance_filt") <= assoc_cutoff_pred)
            )
        )
        .collect()
    )


@timeit
@lazify
def make_graph_based_ids(
    df: pl.DataFrame, graph: nx.Graph, return_cc_list: bool = False
) -> pl.DataFrame:
    cc = list(nx.connected_components(graph))

    cc_df = pl.DataFrame(
        data=[(int(_x), i) for i, x in enumerate(cc) for _x in x],
        schema={
            "object_id": pl.UInt64,
            "vehicle_id": pl.UInt64,
        },
    )

    cc_df = cc_df.vstack(
        df.lazy()
        .select(pl.col("object_id").unique().cast(pl.UInt64))
        .with_context(cc_df.rename({"object_id": "object_id_left"}).lazy())
        .filter(~pl.col("object_id").is_in(pl.col("object_id_left")))
        .with_row_count(
            "vehicle_id",
            offset=cc_df["vehicle_id"][-1] + 1,
        )
        .cast({"vehicle_id": pl.UInt64})
        .select(["object_id", "vehicle_id"])
        .collect()
    )

    df = df.with_columns(pl.col("object_id").cast(pl.UInt64)).join(
        cc_df, on="object_id", how="left"
    )

    if return_cc_list:
        return df, cc
    return df


@timeit
@lazify
def create_vehicle_ids(
    df: pl.DataFrame,
    match_df: pl.DataFrame,
) -> pl.DataFrame:
    import networkx as nx

    g = nx.Graph()

    # create a bidirectional graph of connections, with the weight being the time difference
    for d in match_df.select(
        ["object_id", "leader", "association_distance_filt"]
    ).to_dicts():
        g.add_edge(d["object_id"], d["leader"], weight=d["association_distance_filt"])

    # make the ids
    df, cc = make_graph_based_ids(df, graph=g, return_cc_list=True)
    return (cc, g, df)


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
            # # count the total number of measurements per vehicle
            # count=pl.col("epoch_time").count().over(["epoch_time", "vehicle_id"]),
            timedelta=(pl.col("epoch_time").max() - pl.col("epoch_time"))
            .dt.total_milliseconds()
            .over("vehicle_id"),
        )
        # filter out the first second of measurements if the count > 0
        .filter((pl.col("cumcount") < 1) | (pl.col("cumtime") > 15))
        .collect()
        .lazy()
        # # .drop(["cumtime", "cumcount", "count"])
        .with_columns(
            #  recount
            count=pl.col("epoch_time").count().over(["epoch_time", "vehicle_id"]),
            all_pred=pl.col("prediction").all().over(["epoch_time", "vehicle_id"]),
        )
        .filter(
            ~pl.col("prediction")
            | (pl.col("count") == 1)
            | (pl.col("all_pred") & (pl.col("cumcount") == 1))
        )
        .filter(~(pl.col("all_pred") & (pl.col("timedelta") < prediction_length * 1e3)))
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
