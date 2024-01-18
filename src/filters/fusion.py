from typing import List, Union
from itertools import permutations
import numpy as np
import torch
import polars as pl
from tqdm import tqdm
import pyarrow as pa
from scipy.stats import chi2

from src.filters.vectorized_kalman import (
    build_h_matrix,
    build_r_matrix,
    CALKFilter,
    CVLKFilter,
    CALCFilter,
    pick_device,
)


# def create_z_matrices(df, Z_followers, Z_leaders):
#     positions = ["s", "front_s", "back_s"]
#     # Create all combinations of leader and follower positions
#     for follower_pos, leader_pos in permutations(positions, 2):
#         follower_col = [
#             f"{follower_pos}",
#             "s_velocity",
#             "d",
#             "d_velocity",
#         ]
#         leader_col = [
#             f"{leader_pos}_leader",
#             "s_velocity_leader",
#             "d_leader",
#             "d_velocity_leader",
#         ]

#         # Append follower data to Z_followers
#         Z_followers.append(
#             torch.from_numpy(
#                 df[follower_col]
#                 .to_numpy()
#                 .copy()
#                 .reshape(-1, 4, 1)
#                 .astype(dtype=np.float32),
#             )
#         )

#         # Append leader data to Z_leaders
#         Z_leaders.append(
#             torch.from_numpy(
#                 df[leader_col]
#                 .to_numpy()
#                 .copy()
#                 .reshape(-1, 4, 1)
#                 .astype(dtype=np.float32),
#             )
#         )


def create_z_matrices(df, Z_followers, Z_leaders):
    for pos in ["s", "front_s", "back_s"]:
        for ext, l in zip(["_leader", ""], [Z_leaders, Z_followers]):
            l.append(
                torch.from_numpy(
                    df[
                        [
                            f"{pos}{ext}",
                            f"s_velocity{ext}",
                            f"d{ext}",
                            f"d_velocity{ext}",
                        ]
                    ]
                    .to_numpy()
                    .copy()
                    .reshape(-1, 4, 1)
                    .astype(
                        dtype=np.float32,
                    ),
                )
            )


def loglikelihood(
    df: pl.DataFrame,
    gpu: bool = True,
) -> pl.DataFrame:
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    H = build_h_matrix().to(torch.float32).to(device)

    R = build_r_matrix().to(torch.float32).to(device)

    P_leader = torch.from_numpy(
        df["P_leader"]
        .to_numpy()
        .copy()
        .reshape(-1, 6, 6)
        .astype(
            dtype=np.float32,
        )
    ).to(device)

    S_leader = H @ P_leader @ H.T + R

    Z_followers = []
    Z_leaders = []

    create_z_matrices(df, Z_followers, Z_leaders)

    Z_followers = torch.stack(Z_followers, dim=1).to(device)
    Z_leaders = torch.stack(Z_leaders, dim=1).to(device)

    _log2pi = torch.log(torch.tensor(2 * np.pi, dtype=torch.float32, device=device))

    error = Z_followers - Z_leaders

    vals, vecs = torch.linalg.eigh(S_leader)
    logdet = torch.log(vals).sum(
        axis=-1,
    )
    valsinv = 1 / vals
    U = vecs * valsinv.sqrt().unsqueeze(-1)
    rank = torch.linalg.matrix_rank(U)
    maha = ((U[..., None, :, :] @ error).squeeze() ** 2).sum(axis=-1)
    likelihood, _ = (-0.5 * (_log2pi * rank[..., None] + logdet[..., None] + maha)).max(
        axis=-1
    )
    return df.with_columns(
        [
            pl.Series(likelihood.detach().cpu().numpy()).alias("loglikelihood"),
        ]
    )


def association_loglikelihood_distance(
    df: pl.DataFrame,
    gpu: bool = True,
    dims: int = 4,
) -> pl.DataFrame:
    device = pick_device(gpu)

    H = build_h_matrix().to(device)
    R = build_r_matrix().to(device)

    R[0,0] = 3

    P = torch.from_numpy(
        df["P_leader"]
        .to_numpy()
        .copy()
        .reshape(-1, 6, 6)
        .astype(
            dtype=np.float32,
        )
    ).to(device)

    P += torch.from_numpy(
        df["P"]
        .to_numpy()
        .copy()
        .reshape(-1, 6, 6)
        .astype(
            dtype=np.float32,
        )
    ).to(device)

    S = H @ P @ H.T + R
    S = S[:, :dims, :dims]

    Z_followers = []
    Z_leaders = []

    create_z_matrices(df, Z_followers, Z_leaders)

    # find if any overlap in the z_positions

    Z_followers = torch.stack(Z_followers, dim=1).to(device)[:, :, :dims]
    Z_leaders = torch.stack(Z_leaders, dim=1).to(device)[:, :, :dims]

    Z_error = Z_followers - Z_leaders

    # s_overlap = (
    #     torch.min(
    #         Z_leaders[:, :, 0].max(axis=1)[0], Z_followers[:, :, 0].max(axis=1)[0]
    #     )
    #     - torch.max(
    #         Z_leaders[:, :, 0].min(axis=1)[0], Z_followers[:, :, 0].min(axis=1)[0]
    #     )
    #     > 0
    # ).any(axis=-1)

    d = (
        (
            Z_error.transpose(-1, -2)
            @ torch.linalg.pinv(S, hermitian=True)[..., None, :, :]
            @ Z_error
        ).squeeze()
        + torch.logdet(S)[..., None]
        # + dims * torch.log(torch.tensor(2 * np.pi, dtype=torch.float32, device=device))
        # + two_pi
    ).sqrt()

    d, _ = d.min(axis=-1)
    d[
        torch.isnan(d)
    ] = 1  # replace nan with 1 (this happens when the determinant is near 0)

    return df.with_columns(
        [
            pl.Series(d.detach().cpu().numpy()).alias("association_distance"),
        ]
    )


def mahalanobis_distance(
    df: pl.DataFrame,
    cutoff: float,
    gpu: bool = True,
) -> pl.DataFrame:
    device = torch.device("cuda" if gpu else "cpu")

    H = build_h_matrix().to(device)

    R = build_r_matrix().to(device)

    P_follower = torch.from_numpy(
        df["P"]
        .to_numpy()
        .copy()
        .reshape(-1, 6, 6)
        .astype(
            dtype=np.float32,
        )
    ).to(device)

    P_leader = torch.from_numpy(
        df["P_leader"]
        .to_numpy()
        .copy()
        .reshape(-1, 6, 6)
        .astype(
            dtype=np.float32,
        )
    ).to(device)

    S_leader = H @ P_leader @ H.T + R
    S_follower = H @ P_follower @ H.T + R

    Z_followers = []
    Z_leaders = []

    create_z_matrices(df, Z_followers, Z_leaders)

    Z_followers = torch.stack(Z_followers, dim=1).to(device)
    Z_leaders = torch.stack(Z_leaders, dim=1).to(device)
    error = Z_followers - Z_leaders

    # calculate the mahalanobis distance
    m_sq = (
        error.transpose(-1, -2) @ torch.inverse(S_leader)[:, None, :, :] @ error
    ).squeeze()
    m_sq_other = (
        (-1 * error.transpose(-1, -2))
        @ torch.inverse(S_follower)[:, None, :, :]
        @ (-1 * error)
    ).squeeze()
    # find which is inside the gate
    inside_gate_1 = m_sq < cutoff
    inside_gate_2 = m_sq_other < cutoff
    inside_gate = (inside_gate_1 & inside_gate_2).any(-1)

    return df.with_columns(
        [
            pl.Series(inside_gate.detach().cpu().numpy()).alias("inside_gate"),
        ]
    )


class IMF:
    def __init__(self, df: pl.DataFrame, gpu: bool = True, s_col: str = "s") -> None:
        # create dimensions of everything
        self.t_dim = df["time_index"].max() + 1
        self.v_dim = df["vehicle_id_int"].max() + 1
        self.z_dim = 3
        # pos_dim = 3
        self.x_dim = 6

        Dt = torch.zeros((self.t_dim, self.v_dim, self.z_dim), dtype=torch.float32)
        X = torch.zeros(
            (self.t_dim, self.v_dim, self.z_dim, self.x_dim), dtype=torch.float32
        )
        P = torch.zeros(
            (self.t_dim, self.v_dim, self.z_dim, self.x_dim, self.x_dim),
            dtype=torch.float32,
        )
        # get the indexes and write the data
        t_index = df["time_index"].cast(int).to_numpy(writable=True)
        v_index = df["vehicle_id_int"].cast(int).to_numpy(writable=True)
        z_index = df["vehicle_time_index_int"].cast(int).to_numpy(writable=True)

        # create_z_matrices()

        X[t_index, v_index, z_index] = torch.from_numpy(
            df[
                [
                    s_col,
                    "s_velocity",
                    "s_accel",
                    "d",
                    "d_velocity",
                    "d_accel",
                ]
            ]
            .cast(pl.Float32)
            .to_numpy()
        )

        P[t_index, v_index, z_index] = torch.from_numpy(
            df["P"].to_numpy().copy().reshape(-1, 6, 6).astype(np.float32)
        )

        # adding additional process noise to the length of the vehicle
        # P[t_index, v_index, z_index, 0, 0] += (
        #     torch.from_numpy(df["length_s"].to_numpy().reshape(-1).astype(np.float32))
        #     /
        # )

        # add more process noise to the velocity
        # P[t_index, v_index, z_index, 1, 1] += 1

        Dt[t_index, v_index, z_index] = torch.from_numpy(
            df["time_diff"].to_numpy().astype(np.float32)
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and gpu else "cpu"
        )

        # send all the tensors to CUDA
        self.Dt = Dt.to(self.device)
        self.X = X.to(self.device)
        self.P = P.to(self.device)

        self.X_hat = self.X[..., 0, :].clone()
        self.P_hat = self.P[..., 0, :, :].clone()

    def apply_filter(
        self,
    ) -> None:
        last_mask = torch.ones((self.v_dim, self.z_dim), dtype=torch.bool).to(
            self.device
        )

        for t in tqdm(range(1, self.t_dim)):
            x_t = self.X[t]
            p_t = self.P[t]

            F = CALCFilter.F_static(
                dt_vect=self.Dt[t],
                shape=(self.v_dim, self.z_dim, self.x_dim, self.x_dim),
            ).to(self.device)

            Q = CALCFilter.Q_static(
                dt_vect=self.Dt[t],
                shape=(self.v_dim, self.z_dim, self.x_dim, self.x_dim),
            ).to(self.device)

            x_hat_t_t1 = (F[..., 0, :, :] @ self.X_hat[t - 1,].unsqueeze(-1)).squeeze()

            p_hat_t_t1 = (
                F[..., 0, :, :] @ self.P_hat[t - 1,] @ F[..., 0, :, :].transpose(-1, -2)
                # + Q[..., 0, :, :]
            )

            # predict the individual measurements
            x_t_t1 = (F @ self.X[t - 1].unsqueeze(-1)).squeeze()
            p_t_t1 = F @ self.P[t - 1] @ F.transpose(-1, -2)  # + Q

            # precompute the inverse
            p_hat_t_t1_i = torch.pinverse(p_hat_t_t1)
            p_t_i = torch.pinverse(p_t)
            p_t_t1_i = torch.pinverse(p_t_t1)

            inner_p = torch.zeros(
                (self.v_dim, self.x_dim, self.x_dim), dtype=torch.float32
            ).to(self.device)
            inner_x = torch.zeros((self.v_dim, self.x_dim, 1), dtype=torch.float32).to(
                self.device
            )

            for z in range(self.z_dim):
                # find vehicles that can be used to update
                mask = (x_t[:, z].abs().sum(axis=-1) > 0.1) & last_mask[:, z]

                # save the mask for the next iteration
                last_mask[:, z] = mask.clone()

                inner_p[mask] += p_t_i[mask, z] - p_t_t1_i[mask, z]

                inner_x[mask] += (p_t_i[mask, z] @ x_t[mask, z, :].unsqueeze(-1)) - (
                    p_t_t1_i[mask, z] @ x_t_t1[mask, z].unsqueeze(-1)
                )

            p_main_inv = p_hat_t_t1_i + inner_p
            p_hat_t_t = torch.pinverse(p_main_inv)
            x_hat_t_t = p_hat_t_t @ (p_hat_t_t1_i @ x_hat_t_t1.unsqueeze(-1) + inner_x)

            self.X_hat[t] = x_hat_t_t.squeeze()
            self.P_hat[t] = p_hat_t_t

    def to_df(self) -> pl.DataFrame:
        t_index = np.repeat(np.arange(self.t_dim), self.v_dim)
        v_index = np.tile(np.arange(self.v_dim), self.t_dim)

        X = self.X_hat.detach().cpu().numpy().reshape(-1, 6)
        P = self.P_hat.detach().cpu().numpy().reshape(-1, 6, 6)

        return pl.DataFrame(
            {
                "time_index": t_index,
                "vehicle_id_int": v_index,
                **{
                    f"ci_{dim}": X[:, j]
                    for j, dim in enumerate(
                        ["s", "s_velocity", "s_accel", "d", "d_velocity", "d_accel"]
                    )
                },
                "ci_P": pa.FixedSizeListArray.from_arrays(
                    P.reshape(-1), self.x_dim * self.x_dim
                ),
            }
        ).with_columns(
            pl.col("time_index").cast(pl.UInt32),
            pl.col("vehicle_id_int").cast(pl.UInt32),
            pl.lit(0).cast(pl.UInt32).alias("vehicle_time_index_int"),
        )

    def cleanup(
        self,
    ) -> None:
        # detach from all tensors
        # iterate through variables in the class and del if it is a tensor
        for attr in dir(self):
            if isinstance(getattr(self, attr), torch.Tensor):
                delattr(self, attr)

        # clear the cache
        torch.cuda.empty_cache()


def trace(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.diagonal(dim1=-2, dim2=-1).sum(-1)


def determinant(tesnor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.det(tesnor)


class CI(IMF):
    def __init__(self, df: pl.DataFrame, *args, **kwargs) -> None:
        super().__init__(df, *args, **kwargs)

    def apply_filter(self) -> None:
        # CALKFilter.w_d = 1
        # CALKFilter.w_s = 2
        # CALKFilter.w_s = 0.1

        R = build_r_matrix().to(self.device)
        H = build_h_matrix().to(self.device)

        for t in tqdm(range(1, self.t_dim)):
            x_t = self.X[t]
            p_t = self.P[
                t
            ]  # + torch.eye(self.x_dim, dtype=torch.float32).to(self.device)

            F = CALCFilter.F_static(
                dt_vect=self.Dt[t, :, 0],
                shape=(self.v_dim, self.x_dim, self.x_dim),
            ).to(self.device)

            Q = CALCFilter.Q_static(
                dt_vect=self.Dt[t, :, 0],
                shape=(self.v_dim, self.x_dim, self.x_dim),
            ).to(self.device)

            self.X_hat[t] = (F @ self.X_hat[t - 1,].unsqueeze(-1)).squeeze()
            self.P_hat[t] = F @ self.P_hat[t - 1,] @ F.transpose(-1, -2) + Q
            # do the recursive update
            for z in range(self.z_dim):
                mask = x_t[:, z].abs().sum(axis=-1) > 0.1

                # if z > 0:
                #     # only gate measurements that are > index 0
                #     S = H @ self.P_hat[t, mask] @ H.T + R
                #     z_error = ((x_t[mask, z, :] - self.X_hat[t, mask]) @ H.T).unsqueeze(
                #         -1
                #     )
                #     # calculate the maha distance and gate it
                #     m_sq = (
                #         z_error.transpose(-1, -2) @ torch.pinverse(S) @ z_error
                #     ).squeeze()

                #     # IDK if more expensive to clone the mask
                #     # or to do the indexing. Going with Clone cause lazy
                #     mask[mask.clone()] &= m_sq < chi2.ppf(0.99, 4)

                omega = determinant(p_t[mask, z]) / (
                    determinant(self.P_hat[t, mask]) + determinant(p_t[mask, z])
                )
                # clip omega to (0, 1)

                # a1 = torch.linalg.lstsq()

                a1 = omega[..., None, None] * torch.linalg.pinv(
                    self.P_hat[t, mask], hermitian=True
                )
                a2 = (1 - omega[..., None, None]) * torch.linalg.pinv(
                    p_t[mask, z], hermitian=True
                )

                self.P_hat[t, mask] = torch.linalg.pinv(a1 + a2, hermitian=True)
                self.X_hat[t, mask] = (
                    self.P_hat[t, mask]
                    @ (
                        a1 @ self.X_hat[t, mask].unsqueeze(-1)
                        + a2 @ x_t[mask, z].unsqueeze(-1)
                    )
                ).squeeze()


class ImprovedFastCI(IMF):
    #  from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1591849

    def apply_filter(self) -> None:
        for t in tqdm(range(1, self.t_dim)):
            x_t = self.X[t]
            p_t = self.P[t]

            F = CALCFilter.F_static(
                dt_vect=self.Dt[t, :, 0],
                shape=(self.v_dim, self.x_dim, self.x_dim),
            ).to(self.device)

            Q = CALCFilter.Q_static(
                dt_vect=self.Dt[t, :, 0],
                shape=(self.v_dim, self.x_dim, self.x_dim),
            ).to(self.device)

            self.X_hat[t] = (F @ self.X_hat[t - 1,].unsqueeze(-1)).squeeze()
            self.P_hat[t] = F @ self.P_hat[t - 1,] @ F.transpose(-1, -2) + Q

            for z in range(self.z_dim):
                mask = x_t[:, z].abs().sum(axis=-1) > 0.1

                I_i = torch.linalg.pinv(p_t[mask, z], hermitian=True)
                I_hat = torch.linalg.pinv(self.P_hat[t, mask], hermitian=True)
                d_i = determinant(I_hat + I_i)
                omega = (d_i - determinant(I_i) + determinant(I_hat)) / (2 * d_i)

                a1 = omega[..., None, None] * I_hat
                a2 = (1 - omega[..., None, None]) * I_i

                self.P_hat[t, mask] = torch.linalg.pinv(a1 + a2, hermitian=True)
                self.X_hat[t, mask] = (
                    self.P_hat[t, mask]
                    @ (
                        a1 @ self.X_hat[t, mask].unsqueeze(-1)
                        + a2 @ x_t[mask, z].unsqueeze(-1)
                    )
                ).squeeze()


def batch_join(
    df: pl.DataFrame,
    method: Union[str, object],
    batch_size: int = 8_000,
    gpu: bool = True,
    s_col: str = "s",
) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("vehicle_time_index_int").max().over("vehicle_id") >= 1).alias("filter")
    )

    filter_df = df.filter(pl.col("filter")).clone()

    filter_df = filter_df.join(
        filter_df.select(pl.col("vehicle_id").unique()).with_row_count(
            "vehicle_id_int"
        ),
        on=["vehicle_id"],
    )

    try:
        filter_cls: Union[IMF, CI] = (
            globals()[method] if isinstance(method, str) else method
        )

        dfs = []

        # create a list of the chunks
        for chunk_df in (
            filter_df
            # .filter(pl.col("filter"))
            .with_columns(
                (pl.col("vehicle_id_int") // batch_size).alias("chunk")
            ).partition_by("chunk")
        ):
            offset = chunk_df["vehicle_id_int"].min()

            # create a new vehicle id
            chunk_df = chunk_df.with_columns(
                (pl.col("vehicle_id_int") - offset).alias("vehicle_id_int")
            )

            imf: Union[IMF, CI] = filter_cls(chunk_df, gpu=gpu, s_col=s_col)
            imf.apply_filter()

            dfs.append(
                imf.to_df()
                .rename(
                    {"ci_s": f"ci_{s_col}"},
                )
                .join(
                    chunk_df.select(
                        [
                            "vehicle_id_int",
                            "time_index",
                            "vehicle_time_index_int",
                            "epoch_time",
                            "vehicle_id",
                            "lane",
                            "lane_index",
                        ]
                    ),
                    on=[
                        "time_index",
                        "vehicle_id_int",
                        "vehicle_time_index_int",
                    ],
                    how="inner",
                )
                # .with_columns(
                #     (pl.col("vehicle_id_int") + offset).alias("vehicle_id_int")
                # )
                .drop("vehicle_id_int")
            )

            imf.cleanup()

        non_filter_df = (
            df.filter(~pl.col("filter"))
            .clone()
            .with_columns(
                pl.col(
                    [
                        s_col,
                        "s_velocity",
                        "s_accel",
                        "d",
                        "d_velocity",
                        "d_accel",
                        "P",
                    ]
                ).map_alias(
                    lambda x: f"ci_{x.replace('_filt', '').replace('_CALK', '')}"
                ),
                pl.lit(0).cast(pl.UInt32).alias("chunk"),
            )
            .select(dfs[0].columns)
            .with_columns(
                # these datatypes got me f'd up
                pl.col(f"ci_{s_col}").cast(pl.Float32),
            )
        )

        return pl.concat([*dfs, non_filter_df]).drop(["chunk", "time_diff"])
    except Exception as e:
        if "imf" in locals():
            imf.cleanup()

        raise e


def rts_smooth(
    df: pl.DataFrame, gpu: bool = True, batch_size: int = 20_000, s_col: str = "s"
) -> pl.DataFrame:
    # this one is easy
    df = (
        df.join(
            df.select(pl.col("vehicle_id").unique()).with_row_count("vehicle_id_int"),
            on=["vehicle_id"],
        )
        .sort("epoch_time")
        .with_columns(
            pl.col("epoch_time").cumcount().over("vehicle_id").alias("time_index"),
            (pl.col("epoch_time").diff() / 1000)
            .cast(float)
            .fill_null(0)
            .over("vehicle_id")
            .alias("time_diff"),
        )
    )

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    dfs = []

    # batch again
    for chunk_df in df.with_columns(
        (pl.col("vehicle_id_int") // batch_size).alias("chunk")
    ).partition_by("chunk"):
        _inner_rts(dfs, chunk_df, device, s_col)

        torch.cuda.empty_cache()

    return pl.concat(dfs)


def _inner_rts(
    dfs: List[pl.DataFrame], chunk_df: pl.DataFrame, device: torch.device, s_col: str
) -> pl.DataFrame:
    chunk_df = chunk_df.with_columns(
        (pl.col("vehicle_id_int") - chunk_df["vehicle_id_int"].min()).alias(
            "vehicle_id_int"
        )
    )

    t_dim = chunk_df["time_index"].max() + 1
    v_dim = chunk_df["vehicle_id_int"].max() + 1

    t_inds = chunk_df["time_index"].cast(int).to_numpy()
    v_inds = chunk_df["vehicle_id_int"].cast(int).to_numpy()

    X = torch.zeros(
        (t_dim, v_dim, 6),
        dtype=torch.float32,
    ).to(device)

    x_block = (
        chunk_df[
            [
                f"ci_{s_col}",
                "ci_s_velocity",
                "ci_s_accel",
                "ci_d",
                "ci_d_velocity",
                "ci_d_accel",
            ]
        ]
        .cast(pl.Float32)
        .to_numpy(use_pyarrow=True)
    )

    X[t_inds, v_inds] = torch.from_numpy(x_block.copy()).to(device)

    P = torch.zeros(
        (t_dim, v_dim, 6, 6),
        dtype=torch.float32,
    ).to(device)

    P[t_inds, v_inds] = torch.from_numpy(
        chunk_df["ci_P"].to_numpy().copy().reshape(-1, 6, 6).astype(np.float32)
    ).to(device)

    Dts = torch.zeros((t_dim, v_dim), dtype=torch.float32).to(device)

    Dts[t_inds, v_inds] = torch.from_numpy(
        chunk_df["time_diff"].to_numpy().copy().astype(np.float32)
    ).to(device)

    X_smooth, _ = CALCFilter.rts_smoother(
        X,
        P,
        Dts,
        device=device,
    )
    X_smooth = X_smooth[t_inds, v_inds].detach().cpu().numpy()

    dfs.append(
        chunk_df.with_columns(
            *(
                pl.Series(
                    name=f"{col}_smooth",
                    values=X_smooth[:, j],
                )
                for j, col in enumerate(
                    [s_col, "s_velocity", "s_accel", "d", "d_velocity", "d_accel"]
                )
            )
        )
    )
