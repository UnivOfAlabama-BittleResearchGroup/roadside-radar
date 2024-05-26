import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler
import fastdtw


class TrajectoryScorer:
    def __init__(self, df, label):
        self.df = df
        self.label = label
        self.scaler = StandardScaler()

    # def score_trajectory(self, df, velo):
    #     r_norm, velo_norm = self.normalize(
    #         df[['f32_velocityInDir_mps']].values, velo
    #     )
    #     return fastdtw.dtw(r_norm, velo_norm)[0]

    def score_trajectory(self, df, search_traj):
        return fastdtw.dtw(
            df[["utm_x", "utm_y", "f32_velocityInDir_mps"]].values, search_traj
        )[0]

    def get_scores(self, min_time, max_time, search_trajectory):
        keep_ids = self.df.filter(pl.col("epoch_time").is_between(min_time, max_time))[
            "object_id"
        ].unique()

        traj_database = self.df.filter(pl.col("object_id").is_in(keep_ids)).to_pandas()

        self.scaler.fit(
            np.concatenate(
                (
                    traj_database[["utm_x", "utm_y", "f32_velocityInDir_mps"]].values,
                    search_trajectory,
                )
            )
        )

        search_database = self.scaler.transform(
            traj_database[["utm_x", "utm_y", "f32_velocityInDir_mps"]].values
        )
        search_dataframe = pd.DataFrame(
            search_database, columns=["utm_x", "utm_y", "f32_velocityInDir_mps"]
        )
        # add the object id to the search trajectory
        search_dataframe["object_id"] = traj_database["object_id"].copy()

        search_trajectory = self.scaler.transform(search_trajectory)

        scores = search_dataframe.groupby("object_id").apply(
            lambda x: self.score_trajectory(x, search_trajectory)
        )

        return scores.sort_values()
