import gc
from typing import List, Union
import numpy as np
import torch
import polars as pl
from tqdm import tqdm
import pyarrow as pa

from src.filters.metrics import build_h_matrix, build_r_matrix
from src.filters.vectorized_kalman import (
    CALCFilter,
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
                    # "s",
                    # "front_s",
                    # "back_s",
                    "s",
                    "s_velocity",
                    "s_accel",
                    "d",
                    "d_velocity",
                    "d_accel",
                    # "distanceToFront_s",
                    # "distanceToBack_s",
                ]
            ]
            .cast(pl.Float32)
            .to_numpy()
        )

        P[t_index, v_index, z_index] = torch.from_numpy(
            df["P_CALC"].to_numpy().copy().reshape(-1, 6, 6).astype(np.float32)
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

            x_hat_t_t1 = (F[..., 0, :, :] @ self.X_hat[t - 1,].unsqueeze(-1)).squeeze()

            p_hat_t_t1 = (
                F[..., 0, :, :] @ self.P_hat[t - 1,] @ F[..., 0, :, :].transpose(-1, -2)
                # + Q[..., 0, :, :]
            )

            # predict the individual measurements
            x_t_t1 = (F @ self.X[t - 1].unsqueeze(-1)).squeeze()
            p_t_t1 = F @ self.P[t - 1] @ F.transpose(-1, -2)  # + Q

            # precompute the inverse
            p_hat_t_t1_i = torch.linalg.pinv(p_hat_t_t1, hermitian=True)
            p_t_i = torch.linalg.pinv(p_t, hermitian=True)
            p_t_t1_i = torch.linalg.pinv(p_t_t1, hermitian=True)

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
            p_hat_t_t = torch.linalg.pinv(p_main_inv, hermitian=True)
            x_hat_t_t = p_hat_t_t @ (p_hat_t_t1_i @ x_hat_t_t1.unsqueeze(-1) + inner_x)

            self.X_hat[t] = x_hat_t_t.squeeze()
            self.P_hat[t] = p_hat_t_t

    def to_df(self) -> pl.DataFrame:
        t_index = np.repeat(np.arange(self.t_dim), self.v_dim)
        v_index = np.tile(np.arange(self.v_dim), self.t_dim)

        X = self.X_hat.cpu().numpy().reshape(-1, 6)
        P = self.P_hat.cpu().numpy().reshape(-1, 6, 6)

        del self.X_hat
        del self.P_hat

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
                setattr(self, attr, torch.zeros(1))

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
            # for z in range(self.z_dim, ):
            # update in reverse order
            for z in range(self.z_dim):
                z = self.z_dim - z - 1

                mask = x_t[:, z].abs().sum(axis=-1) > 0.1

                omega = determinant(p_t[mask, z]) / (
                    determinant(self.P_hat[t, mask]) + determinant(p_t[mask, z])
                )

                a1 = omega[..., None, None] * torch.linalg.pinv(
                    self.P_hat[t, mask],
                )
                a2 = (1 - omega[..., None, None]) * torch.linalg.pinv(
                    p_t[mask, z],
                )

                self.P_hat[t, mask] = torch.linalg.pinv(
                    a1 + a2,
                )
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
        R = build_r_matrix(
            pos_error=np.sqrt(0.5),
            pos_velo_error=np.sqrt(0.8112),
            d_pos_error=np.sqrt(0.1),
            d_velo_error=np.sqrt(0.1),
            dims=4,
        ).to(self.device)
        H = build_h_matrix(dims=4).to(self.device)

        P_mod = (
            H.T @ R @ H
        )  # / 10 #+ CALCFilter.P_mod * torch.eye(self.x_dim).to(self.device)
        self.P += P_mod
        self.P_hat += P_mod

        for t in tqdm(range(1, self.t_dim)):
            x_t = self.X[t]
            p_t = self.P[t]

            F = CALCFilter.F_static(
                dt_vect=self.Dt[t, :, 0],
                shape=(self.v_dim, self.x_dim, self.x_dim),
                # duplicate_pos_count=3,
            ).to(self.device)

            Q = CALCFilter.Q_static(
                dt_vect=self.Dt[t, :, 0],
                shape=(self.v_dim, self.x_dim, self.x_dim),
                # duplicate_pos_count=3
            ).to(self.device)
            self.X_hat[t] = (F @ self.X_hat[t - 1,].unsqueeze(-1)).squeeze()
            self.P_hat[t] = F @ self.P_hat[t - 1,] @ F.transpose(-1, -2) + Q

            for z in range(self.z_dim):
                mask = x_t[:, z].abs().sum(axis=-1) > 0.1

                I_i = torch.linalg.pinv(
                    p_t[mask, z],
                )
                I_hat = torch.linalg.pinv(
                    self.P_hat[t, mask],
                )
                d_i = determinant(I_hat + I_i)
                omega = (d_i - determinant(I_i) + determinant(I_hat)) / (2 * d_i)

                a1 = omega[..., None, None] * I_hat
                a2 = (1 - omega[..., None, None]) * I_i

                self.P_hat[t, mask] = torch.linalg.pinv(
                    a1 + a2,
                )
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
    batch_size: int = 1_000_000,
    gpu: bool = True,
    s_col: str = "s",
) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("vehicle_time_index_int").max().over("vehicle_id") >= 1).alias("filter")
    )

    filter_df = df.filter(pl.col("filter")).lazy()

    filter_df = filter_df.join(
        filter_df.select(pl.col("vehicle_id").unique()).with_row_count(
            "vehicle_id_int"
        ),
        on=["vehicle_id"],
    ).collect(streaming=True)

    try:
        filter_cls: Union[IMF, CI] = (
            globals()[method] if isinstance(method, str) else method
        )

        dfs = []

        # create a list of the chunks
        for chunk_df in filter_df.with_columns(
            (pl.col("vehicle_id_int") // batch_size).alias("chunk")
        ).partition_by("chunk"):
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
                .drop("vehicle_id_int")
            )

            # try to free GPU memory
            imf.cleanup()
            del imf
            torch.cuda.empty_cache()
            gc.collect()

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
            imf.cleanup()  # noqa: F821
        del imf  # noqa: F821
        torch.cuda.empty_cache()
        gc.collect()

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
        chunk_df.select(
            pl.concat_list(
                [
                    f"ci_{s_col}",
                    "ci_s_velocity",
                    "ci_s_accel",
                    "ci_d",
                    "ci_d_velocity",
                    "ci_d_accel",
                ]
            )
            .list.to_array(width=6)
            .alias("ci_x")
        )["ci_x"]
        .to_numpy(writable=True)
        .copy()
    )

    X[t_inds, v_inds] = torch.from_numpy(x_block.copy()).to(device)

    P = torch.zeros(
        (t_dim, v_dim, 6, 6),
        dtype=torch.float32,
    ).to(device)

    P[t_inds, v_inds] = torch.from_numpy(
        chunk_df["ci_P"].to_numpy().copy().reshape(-1, 6, 6).astype(np.float32)
    ).to(device)

    # P[..., 1, 1] += 2

    Dts = torch.zeros((t_dim, v_dim), dtype=torch.float32).to(device)

    Dts[t_inds, v_inds] = torch.from_numpy(
        chunk_df["time_diff"].to_numpy().copy().astype(np.float32)
    ).to(device)

    # CALCFilter.w_s = 1

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
