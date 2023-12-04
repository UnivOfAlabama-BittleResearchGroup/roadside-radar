from typing import Union
import polars as pl
import networkx as nx


def _safe_lazy(df: Union[pl.LazyFrame, pl.DataFrame]) -> pl.DataFrame:
    return df if isinstance(df, pl.LazyFrame) else df.lazy()


def _safe_collect(df: Union[pl.LazyFrame, pl.DataFrame]) -> pl.DataFrame:
    return df.collect() if isinstance(df, pl.LazyFrame) else df


def build_match_df(
    df: pl.DataFrame,
    s_thresh: float = 10,
    object_id_col: str = "object_id",
    s_col: str = "s_centroid",
) -> pl.DataFrame:
    lazy_df = _safe_lazy(df)

    matching_df = (
        lazy_df.sort(["epoch_time", s_col], descending=[False, True])
        .set_sorted(["epoch_time", s_col])
        .with_row_count("index")
    )

    return (
        matching_df.with_columns(
            [
                pl.col(object_id_col)
                .shift(-1)
                .over(["epoch_time", "lane_hash"])
                .alias("leader"),
            ]
        )
        .join(
            matching_df,
            left_on=[
                "leader",
                "epoch_time",
            ],
            right_on=[
                object_id_col,
                "epoch_time",
            ],
            how="inner",
            suffix="_leader",
        )
        .sort("epoch_time")
        # calculate the s gap
        .with_columns(
            (pl.col(f"{s_col}_leader") - pl.col(s_col)).alias("s_gap"),
        )
        # filter out data points where both are prediction
        .pipe(_safe_collect)
    )


def create_vehicle_ids(
    df: pl.DataFrame,
    match_df: pl.DataFrame,
) -> pl.DataFrame:
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


def calculate_match_indexes(
    df: pl.DataFrame,
) -> pl.DataFrame:
    return (
        _safe_lazy(df)
        .with_columns(
            # sort the object id and leader
            pl.struct(
                [
                    pl.col("leader"),
                    pl.col("object_id"),
                ]
            ).alias("pair")
        )
        .with_columns(
            (pl.col("prediction") | pl.col("prediction_leader")).alias("prediction_any")
        )
        # create a pair object that contains the object id and the leader (sorted)
        .with_columns(
            [
                # ----------- Calculate the Time Headway ------------
                (
                    (pl.col("s_centroid_leader") - pl.col("s_centroid"))
                    / pl.col("s_velocity_filt")
                ).alias("headway"),
                # ----------- Calculate the Match Indexes ------------
                # the first method is to find the median and then take the 10 before and after
                # the reason is that starts and ends have a lot of noise
                # ---------------------------
                # find the start index to take
                # ((pl.count().cast(pl.Int32) // 2) - 20)
                pl.lit(0)
                # .clip_min(0)
                # .over("pair")
                .alias("start_index"),
                # find the end index to use
                pl.when(
                    # ((pl.count().cast(pl.Int32) // 2) + 20) < pl.count(),
                    pl.count()
                    > 10,
                )
                .then(10)
                .otherwise(pl.count())
                # pl.count()
                .over("pair").alias("end_index"),
                # -------- Calculate the Match Indexes Take 2 ------------
                # Find periods where there is no prediction
                # prediction naturally has more uncertainty
                # ---------------------------
                # find the start index to take
                pl.col("object_id")
                .cumcount()
                .filter(~pl.col("prediction_any"))
                .first()  # skip the first one
                .over("pair")
                .fill_null(0)
                .alias("start_index_no_pred"),
                # find the end index to use
                pl.col("object_id")
                .cumcount()
                .filter(~pl.col("prediction_any"))
                .last()
                .over("pair")
                .fill_null(0)
                .alias("end_index_no_pred"),
            ]
        )
        .pipe(_safe_collect)
    )


def pipe_gate_headway_calc(
    df: pl.DataFrame,
    alpha: float = 0.2,
) -> pl.DataFrame:
    return (
        _safe_lazy(df)
        .group_by(
            "pair",
        )
        .agg(
            # find the headway in the middle of the leader-follower pair
            pl.when((pl.col("end_index_no_pred").first() >= 10))
            .then(
                pl.col("association_distance")
                .slice(
                    pl.col("start_index_no_pred").first(),
                    pl.col("end_index_no_pred").first(),
                )
                .ewm_mean(alpha=alpha)
                .last()
            )
            .otherwise(
                pl.col("association_distance")
                .sort_by(
                    [
                        pl.col("prediction_any"),
                        pl.col("epoch_time"),
                    ]
                )
                .ewm_mean(alpha=alpha)
                .last()
            ).alias("association_distance_filt"),
            # pl.when((pl.col("end_index_no_pred").first() >= 10))
            # .then(
            #     (
            #         pl.col("s_gap")
            #         .slice(
            #             pl.col("start_index_no_pred").first(),
            #             pl.col("end_index_no_pred").first(),
            #         )
            #         .abs()
            #         .max()
            #     )
            # )
            # .otherwise(
            #     (
            #         pl.col("s_gap")
            #         .slice(pl.col("start_index").first(), pl.col("end_index").first())
            #         .abs()
            #         .max()
            #     )
            # )
            # .alias("s_gap"),
            pl.col("epoch_time").first().alias("epoch_time"),
            pl.col("prediction").any().alias("prediction"),
            pl.col("prediction_leader").any().alias("prediction_leader"),
            # pl.col('s_gap').abs().max()
            # pl.when(pl.col("prediction").all() | pl.col("prediction_leader").all())
            # .then(pl.col("association_distance").reverse().ewm_mean(alpha=alpha).last())
            # .otherwise(
            #     pl.col("association_distance")
            #     .sort_by(
            #         [
            #             pl.col("prediction"),
            #             pl.col("prediction_leader"),
            #             pl.col("epoch_time"),
            #         ]
            #     )
            #     .ewm_mean(alpha=alpha)
            #     .last()
            # )
            # .alias("association_distance_filt"),
        )
        .pipe(_safe_collect)
    )
