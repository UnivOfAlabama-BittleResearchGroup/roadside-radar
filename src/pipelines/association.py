from datetime import timedelta
from itertools import combinations
from typing import List
import numpy as np
import polars as pl
from src.filters.fusion import association_loglikelihood_distance
from src.pipelines.utils import lazify, timeit
from scipy.stats import chi2
import networkx as nx
import heapq


@lazify
@timeit
def add_front_back_s(
    df: pl.DataFrame,
    use_median_length: bool = False,
    use_nearest_length: bool = False,
    use_global_median: bool = False,
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

    if use_global_median:
        median_length = df.select(
            pl.col("length_s").filter(pl.col("dist").is_between(10, 30)).median()
        ).collect()["length_s"][0]

        return (
            df.with_columns(
                pl.when((pl.col("dist") >= 30) & (pl.col("length_s") < median_length))
                .then(pl.lit(median_length))
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


@timeit
@lazify
def calc_assoc_liklihood_distance(
    df: pl.DataFrame,
    gpu: bool = True,
    batch_size: int = 100_000,
    dims: int = 4,
    permute: bool = False,
) -> pl.DataFrame:
    import torch

    dfs = []

    for _df in (
        df.with_row_count("maha_ind")
        .with_columns((pl.col("maha_ind") // batch_size).alias("chunk"))
        .partition_by("chunk")
    ):
        dfs.append(association_loglikelihood_distance(_df, gpu, dims, permute))

    try:
        torch.cuda.empty_cache()
    except:  # noqa: E722
        pass

    return pl.concat(dfs).drop(["maha_ind", "chunk"])


@timeit
@lazify
def calculate_match_indexes(
    df: pl.DataFrame,
    min_time_threshold: float = 0.5,
) -> pl.DataFrame:
    return (
        df.lazy()
        .with_columns(
            pl.when(pl.col("leader") < pl.col("object_id"))
            .then(pl.concat_list([pl.col("leader"), pl.col("object_id")]))
            .otherwise(pl.concat_list([pl.col("object_id"), pl.col("leader")]))
            .alias("pair")
        )
        # .with_columns(
        #     (pl.col("prediction") | pl.col("prediction_leader")).alias(
        #         "prediction_any"
        #     ),
        #     (pl.col('pair').cum_count() / 10)  # simpler way to get the "match time"
        #     .over("pair")
        #     .alias("match_time"),
        # )
        # .filter(
        #     (
        #         (~pl.col("prediction_any"))
        #         | (pl.col("match_time") < min_time_threshold)
        #     )
        # )
        .with_columns(
            [
                # ----------- Calculate the Time Headway ------------
                ((pl.col("s_leader") - pl.col("s")) / pl.col("s_velocity")).alias(
                    "headway"
                ),
                # pl.col("object_id").cumcount().over("pair").alias("sort_index"),
            ]
        )
        # .sort(
        #     ["epoch_time"],
        # )
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
            pl.col("association_distance").min().alias("association_distance_filt"),
            pl.col("epoch_time")
            .filter(pl.col("association_distance") < association_dist_cutoff)
            .first()
            .alias("join_time"),
            pl.col("epoch_time").first().alias("epoch_time"),
            pl.col("epoch_time").last().alias("epoch_time_max"),
            pl.col("prediction").any().alias("prediction"),
            pl.col("prediction_leader").any().alias("prediction_leader"),
            # pl.col("headway").ewm_mean(alpha=alpha).mean().alias("headway"),
            pl.col("headway").mean().alias("headway"),
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
    time_headway_cutoff: float,
    # headway_cutoff_inner: float,
) -> pl.DataFrame:
    valid_matches = (
        df
        # .lazy()
        .sort("epoch_time")
        # .unnest("pair")
        .with_columns(
            pl.col("pair").list.get(0).alias("object_id"),
            pl.col("pair").list.get(1).alias("leader"),
        )
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
                "headway",
            ]
        )
        .filter(
            (
                pl.when(~(pl.col("prediction") | pl.col("prediction_leader")))
                .then(pl.col("association_distance_filt") <= assoc_cutoff)
                .otherwise(pl.col("association_distance_filt") <= assoc_cutoff_pred)
                # | (pl.col("headway") < time_headway_cutoff)
            )
        )
        # .sort("epoch_time")
        # .with_columns(
        #     pl.when(pl.col("variable") == "object_id")
        #     .then(pl.col("prediction"))
        #     .otherwise(pl.col("prediction_leader"))
        #     .alias("prediction"),
        #     pl.when(pl.col("variable") == "object_id")
        #     .then(pl.col("epoch_time_max"))
        #     .otherwise(pl.col("epoch_time_max_leader"))
        #     .alias("my_end_time"),
        #     pl.when(pl.col("variable") == "object_id")
        #     .then(pl.col("epoch_time_max_leader"))
        #     .otherwise(pl.col("epoch_time_max"))
        #     .alias("other_end_time"),
        # )
        # .drop(
        #     ["epoch_time_max", "epoch_time_max_leader", "prediction_leader", "variable"]
        # )
        # .with_columns(
        #     pl.col("prediction").cumsum().over("value").alias("prediction_count"),
        #     pl.col("other_end_time")
        #     .filter(~pl.col("prediction"))
        #     .max()
        #     .over("value")
        #     .alias("other_end_time_max"),
        # )
        # # .filter(
        # #     (pl.col("prediction_count") <= 1)
        # #     & (pl.col("row_nr").count().over("row_nr") > 1)
        # # )
        .collect()
    )

    return valid_matches.filter(
        pl.col("row_nr").is_in(keep_rows["row_nr"].unique())
    ).drop("row_nr")
    # return valid_matches


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
        .filter(
            ~(
                (pl.col("all_pred") & (pl.col("timedelta") < prediction_length * 1e3))
                
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


def get_edge_attributes(G, name, cutoff):
    # ...
    edges = G.edges(data=True)
    return (x[-1][name] or (cutoff - 1e-3) for x in edges)


def get_graph_score(
    sub_graph: nx.Graph,
    # df: pl.DataFrame,
    remove_edges: List[int] = (),
    add_edges: List[int] = (),
    reinstate_graph: bool = True,
    big_G: nx.Graph = None,
    cutoff: float = None,
    # combs: List[str] = None,
) -> float:
    for remove in remove_edges:
        sub_graph.remove_edge(*remove)

    for add_edge in add_edges:
        sub_graph.add_edge(*add_edge)

    # return score
    scores = []
    for subgraph in nx.connected_components(sub_graph):
        sg = big_G.subgraph(subgraph)
        try:
            scores.extend(get_edge_attributes(sg, "weight", cutoff))
        except TypeError:
            scores.append(cutoff - 0.0001)

    if reinstate_graph:
        for remove in remove_edges:
            sub_graph.add_edge(*remove)

        for add_edge in add_edges:
            sub_graph.remove_edge(*add_edge)

    return scores


# def check_improvement(old_score, new_score, cutoff):

#     # check that there is improvement over the last score
#     return any(s < cutoff for s in new_score) and np.mean(new_score) < np.mean(old_score)
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1, 0, -1))


def walk_graph_removals(
    g: nx.Graph,
    cutoff: float = chi2.ppf(0.95, 4),
    max_removals: int = 3,
    big_G: nx.Graph = None,
):
    i = 0
    remove_edges = []
    graph_scores = get_graph_score(
        g,
        big_G=big_G,
        cutoff=cutoff,
    )

    if all(s < cutoff for s in graph_scores):
        return []
    
    fresh_g = g.copy()

    # sort the edges by their node connectivity
    edges = list(sorted(g.edges, key=lambda x: g.degree(x[0]) + g.degree(x[1])))
    graph_scores = []
    
    while i == 0 or (
        (i < max_removals) and not all(s < cutoff for s in graph_scores[-1])
    ):
        # for _ in range(max_removals):
        scores = []
        for edge in edges:
            score = get_graph_score(
                # score_func,
                g,
                remove_edges=[edge],
                big_G=big_G,
                cutoff=cutoff,
            )
            # score = np.mean([s for s in score if s is not None])
            # scores.append((score, edge))
            heapq.heappush(scores, (sum(score) / (len(score) or 1), score, edge))

        # # store the scores
        # # keep any that reduce the score underneath the cutoff
        # scores = sorted(scores, key=lambda x: np.mean(x[0]))
        opt_edge = heapq.heappop(scores)

        # check the score
        remove_edges.append(opt_edge[-1])
        graph_scores.append(opt_edge[1])

        # remove what we don't need
        g.remove_edge(*opt_edge[-1])
        edges.remove(opt_edge[-1])

        i += 1

    # now loop and see if we can add any edges back and still satisfy the requirements.
    # This happends for multiple reasons
    # delete_edges = remove_edges.copy()
    pop_edges = []
    # all_valid_edges = set(fresh_g)
    for i, edge in enumerate(remove_edges):
        # edge = list(set(edge).union(all_valid_edges))
        # if not len(edge):
        #     continue
        score = get_graph_score(g, add_edges=[edge], big_G=big_G, cutoff=cutoff)

        if all(s < cutoff for s in score):
            # print("adding_back", edge)
            pop_edges.append(i)
            g.add_edge(*edge)
            # for e in edge:
            # g.add_edge(*e)

    remove_edges = list(nx.difference(fresh_g, g).edges)

    return remove_edges
