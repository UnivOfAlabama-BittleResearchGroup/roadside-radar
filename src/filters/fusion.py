from typing import List, Union
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
    CALCFilter,
    pick_device,
)


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
) -> pl.DataFrame:
    device = pick_device(gpu)

    H = build_h_matrix().to(device)
    R = build_r_matrix().to(device)

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

    Z_followers = []
    Z_leaders = []

    create_z_matrices(df, Z_followers, Z_leaders)

    Z_error = torch.stack(Z_followers, dim=1).to(device) - torch.stack(
        Z_leaders, dim=1
    ).to(device)

    # two_pi = 4 * torch.log(torch.tensor([2 * math.pi], device=device))

    d = (
        (
            Z_error.transpose(-1, -2) @ torch.pinverse(S)[..., None, :, :] @ Z_error
        ).squeeze()
        + torch.log(torch.det(S))[..., None]
        # + two_pi
    ).sqrt()

    d, _ = d.min(axis=-1)

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
    def __init__(self, df: pl.DataFrame, gpu: bool = True) -> None:
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
                    "s",
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
            df["P_CALK"].to_numpy().copy().reshape(-1, 6, 6).astype(np.float32)
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

            F = CALKFilter.F_static(
                dt_vect=self.Dt[t],
                shape=(self.v_dim, self.z_dim, self.x_dim, self.x_dim),
            ).to(self.device)

            Q = CALKFilter.Q_static(
                dt_vect=self.Dt[t],
                shape=(self.v_dim, self.z_dim, self.x_dim, self.x_dim),
            ).to(self.device)

            x_hat_t_t1 = (F[..., 0, :, :] @ self.X_hat[t - 1,].unsqueeze(-1)).squeeze()

            p_hat_t_t1 = (
                F[..., 0, :, :] @ self.P_hat[t - 1,] @ F[..., 0, :, :].transpose(-1, -2)
                + Q[..., 0, :, :]
            )

            # predict the individual measurements
            x_t_t1 = (F @ self.X[t - 1].unsqueeze(-1)).squeeze()
            p_t_t1 = F @ self.P[t - 1] @ F.transpose(-1, -2) + Q

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


def trace(tesnor: torch.Tensor) -> torch.Tensor:
    return tesnor.diagonal(dim1=-2, dim2=-1).sum(-1)


class CI(IMF):
    def __init__(self, df: pl.DataFrame, *args, **kwargs) -> None:
        super().__init__(df, *args, **kwargs)

    def apply_filter(self) -> None:
        # CALKFilter.w_d = 1
        # CALKFilter.w_s = 2

        R = build_r_matrix().to(self.device)
        H = build_h_matrix().to(self.device)

        for t in tqdm(range(1, self.t_dim)):
            x_t = self.X[t]
            p_t = self.P[t]

            F = CALKFilter.F_static(
                dt_vect=self.Dt[t, :, 0],
                shape=(self.v_dim, self.x_dim, self.x_dim),
            ).to(self.device)

            Q = CALKFilter.Q_static(
                dt_vect=self.Dt[t, :, 0],
                shape=(self.v_dim, self.x_dim, self.x_dim),
            ).to(self.device)

            self.X_hat[t] = (F @ self.X_hat[t - 1,].unsqueeze(-1)).squeeze()
            self.P_hat[t] = F @ self.P_hat[t - 1,] @ F.transpose(-1, -2) + Q
            # do the recursive update
            for z in range(self.z_dim):
                mask = x_t[:, z].abs().sum(axis=-1) > 0.1

                if z > 0:
                    # only gate measurements that are > index 0
                    S = H @ self.P_hat[t, mask] @ H.T + R
                    z_error = ((x_t[mask, z, :] - self.X_hat[t, mask]) @ H.T).unsqueeze(
                        -1
                    )
                    # calculate the maha distance and gate it
                    m_sq = (
                        z_error.transpose(-1, -2) @ torch.pinverse(S) @ z_error
                    ).squeeze()

                    # IDK if more expensive to clone the mask
                    # or to do the indexing. Going with Clone cause lazy
                    mask[mask.clone()] &= m_sq < chi2.ppf(0.99, 4)

                omega = trace(p_t[mask, z]) / (
                    trace(self.P_hat[t, mask]) + trace(p_t[mask, z])
                )
                a1 = omega[..., None, None] * torch.pinverse(self.P_hat[t, mask])
                a2 = (1 - omega[..., None, None]) * torch.pinverse(p_t[mask, z])

                self.P_hat[t, mask] = torch.pinverse(a1 + a2)
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
    filter_again: bool = False,
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

            imf: Union[IMF, CI] = filter_cls(chunk_df, gpu=gpu)
            imf.apply_filter()

            dfs.append(
                imf.to_df()
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
                        "s",
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
                pl.col("ci_s").cast(pl.Float32),
            )
        )

        return (
            pl.concat([*dfs, non_filter_df])
            # pl.concat(dfs)
            .drop(["chunk", "time_diff"])
            # .sort(
            #     [
            #         "vehicle_id",
            #         "epoch_time",
            #     ]
            # )
            # .set_sorted(
            #     [
            #         "vehicle_id",
            #         "epoch_time",
            #     ]
            # )
        )
    except Exception as e:
        if "imf" in locals():
            imf.cleanup()

        raise e


def rts_smooth(
    df: pl.DataFrame, gpu: bool = True, batch_size: int = 20_000
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
        _inner_rts(dfs, chunk_df, device)

        torch.cuda.empty_cache()

    return pl.concat(dfs)


def _inner_rts(
    dfs: List[pl.DataFrame], chunk_df: pl.DataFrame, device: torch.device
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

    X[t_inds, v_inds] = torch.from_numpy(
        chunk_df[
            [
                "ci_s",
                "ci_s_velocity",
                "ci_s_accel",
                "ci_d",
                "ci_d_velocity",
                "ci_d_accel",
            ]
        ]
        .cast(pl.Float32)
        .to_numpy()
        .copy()
    ).to(device)

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

    CALKFilter.w_s = 5
    CALKFilter.w_d = 2

    X_smooth, _ = CALKFilter.rts_smoother(
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
                    ["s", "s_velocity", "s_accel", "d", "d_velocity", "d_accel"]
                )
            )
        )
    )
