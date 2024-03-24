import gc
from typing import List, Union
from itertools import permutations
import numpy as np
import torch
import polars as pl
from tqdm import tqdm
import pyarrow as pa
from scipy.stats import chi2

from src.filters.vectorized_kalman import (
    # build_h_matrix,
    # build_r_matrix,
    CALKFilter,
    CVLKFilter,
    CALCFilter,
    pick_device,
    gen_chunks,
)


def create_z_matrices_permute(df, Z_followers, Z_leaders, column_creator):
    positions = ["s", "front_s", "back_s"]
    # Create all combinations of leader and follower positions
    for follower_pos, leader_pos in permutations(positions, 2):
        # Append follower data to Z_followers
        Z_followers.append(
            torch.from_numpy(
                df[column_creator(follower_pos, "")]
                .to_numpy()
                .copy()
                .reshape(-1, 2, 1)
                .astype(dtype=np.float32),
            )
        )

        # Append leader data to Z_leaders
        Z_leaders.append(
            torch.from_numpy(
                df[column_creator(leader_pos, "_leader")]
                .to_numpy()
                .copy()
                .reshape(-1, 2, 1)
                .astype(dtype=np.float32),
            )
        )


def build_h_matrix(dims: int = 4) -> torch.FloatTensor:
    """
    Build the H matrix for the Kalman filter.

    IDK why this is a function, but it is. ChatGPT made me do it
    """
    if dims == 6:
        return torch.FloatTensor(
            np.eye(6),
        )

    elif dims == 4:
        return torch.FloatTensor(
            np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                ]
            ),
        )
    elif dims == 2:
        return torch.FloatTensor(
            np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                ]
            ),
        )


def build_r_matrix(
    pos_error: float = 1.5,
    pos_velo_error: float = 1,
    d_pos_error: float = 1.5,
    d_velo_error: float = 1,
    dims: int = 4,
) -> torch.FloatTensor:
    """
    Build the R matrix for the Kalman filter.

    IDK why this is a function, but it is. ChatGPT made me do it
    """
    if dims == 6:
        return (
            torch.FloatTensor(
                np.diag(
                    [
                        pos_error,
                        pos_velo_error,
                        1,
                        d_pos_error,
                        d_velo_error,
                        1,
                    ]
                ),
            )
            ** 2
        )
    elif dims == 4:
        return (
            torch.FloatTensor(
                np.diag(
                    [
                        pos_error,
                        pos_velo_error,
                        d_pos_error,
                        d_velo_error,
                    ]
                ),
            )
            ** 2
        )
    elif dims == 2:
        return (
            torch.FloatTensor(
                np.diag(
                    [
                        pos_error,
                        d_pos_error,
                    ]
                ),
            )
            ** 2
        )


def create_z_matrices(
    df, Z_followers, Z_leaders, pos_override=None, dims: int = 4, permute: bool = False
):
    def column_creator(pos, ext):
        if dims == 6:
            return [
                f"{pos}{ext}",
                f"s_velocity{ext}",
                f"s_accel{ext}",
                f"d{ext}",
                f"d_velocity{ext}",
                f"d_accel{ext}",
            ]
        if dims == 4:
            return [
                f"{pos}{ext}",
                f"s_velocity{ext}",
                f"d{ext}",
                f"d_velocity{ext}",
            ]
        elif dims == 2:
            return [
                f"{pos}{ext}",
                f"d{ext}",
            ]

    if permute:
        return create_z_matrices_permute(df, Z_followers, Z_leaders, column_creator)

    if pos_override is None:
        pos_override = ["s", "front_s", "back_s"]
    for pos in pos_override:
        for ext, l in zip(["_leader", ""], [Z_leaders, Z_followers]):
            l.append(
                torch.from_numpy(
                    df[column_creator(pos, ext)]
                    .to_numpy()
                    .copy()
                    .reshape(-1, dims, 1)
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

    R = build_r_matrix(pos_error=2.5, d_pos_error=1.5).to(torch.float32).to(device)

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
    # augment_length: bool = True,
    length_cols: List[str] = ["length_s", "length_s_leader"],
    permute: bool = False,
) -> pl.DataFrame:
    device = pick_device(gpu)

    H = build_h_matrix(dims=dims).to(device)
    R = build_r_matrix(d_pos_error=1.5, pos_error=2.5, dims=dims).to(device)
    # R = build_r_matrix().to(device)

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

    # S = R
    S = H @ P @ H.T
    # if dims == 2:
    S += R
    # S = S

    Z_followers = []
    Z_leaders = []

    create_z_matrices(df, Z_followers, Z_leaders, dims=dims, permute=permute)

    # find if any overlap in the z_positions
    Z_followers = torch.stack(Z_followers, dim=1).to(device)
    Z_leaders = torch.stack(Z_leaders, dim=1).to(device)

    Z_error = Z_followers - Z_leaders

    d = (
        (Z_error.transpose(-1, -2) @ S.pinverse()[..., None, :, :] @ Z_error).squeeze()
        # + torch.logdet(S)[..., None]
    )
    # this is the associtiation liklihood bit
    if device.type == "mps":
        # d += torch.logdet(S.cpu())[..., None]
        d = d.detach().cpu() + torch.logdet(S.cpu())[..., None]

    else:
        d += torch.logdet(S)[..., None]
        d = d.detach().cpu()

    # check 2d intersection of the measurements
    # overlap =

    d = d.min(axis=-1)[0].numpy()

    # zero out all local tensors
    del Z_followers
    del Z_leaders
    del Z_error
    del P
    del S
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gc.collect()

    return df.with_columns(
        [
            pl.Series(d).alias("association_distance"),
        ]
    )


# def mahalanobis_distance(
#     df: pl.DataFrame,
#     cutoff: float = None,
#     gpu: bool = True,
#     pos_override=None,
#     return_maha: bool = False,
# ) -> pl.DataFrame:
#     device = torch.device("cuda" if gpu else "cpu")

#     H = build_h_matrix().to(device)

#     R = build_r_matrix().to(device)

#     P_follower = torch.from_numpy(
#         df["P"]
#         .to_numpy()
#         .copy()
#         .reshape(-1, 6, 6)
#         .astype(
#             dtype=np.float32,
#         )
#     ).to(device)

#     P_leader = torch.from_numpy(
#         df["P_leader"]
#         .to_numpy()
#         .copy()
#         .reshape(-1, 6, 6)
#         .astype(
#             dtype=np.float32,
#         )
#     ).to(device)

#     S_leader = H @ P_leader @ H.T + R
#     S_follower = H @ P_follower @ H.T + R

#     S = S_leader + S_follower

#     Z_followers = []
#     Z_leaders = []

#     create_z_matrices(df, Z_followers, Z_leaders, pos_override=pos_override)

#     Z_followers = torch.stack(Z_followers, dim=1).to(device)
#     Z_leaders = torch.stack(Z_leaders, dim=1).to(device)
#     Z_error = Z_followers - Z_leaders

#     # calculate the mahalanobis distance
#     m_sq = (
#         Z_error.transpose(-1, -2)
#         @ torch.linalg.pinv(
#             S,
#         )[..., None, :, :]
#         @ Z_error
#     ).squeeze()

#     if return_maha:
#         return df.with_columns(
#             [
#                 pl.Series(m_sq.detach().cpu().numpy()).alias("m_sq"),
#             ]
#         )

#     assert cutoff is not None, "Must provide a cutoff"

#     # find which is inside the gate
#     inside_gate_1 = m_sq < cutoff

#     inside_gate = (inside_gate_1).any(-1)

#     return df.with_columns(
#         [
#             pl.Series(inside_gate.detach().cpu().numpy()).alias("inside_gate"),
#         ]
#     )


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
            self.P_hat[t] = F @ self.P_hat[t - 1,] @ F.transpose(-1, -2)  # + Q
            # do the recursive update
            # for z in range(self.z_dim, ):
            # update in reverse order
            for z in range(self.z_dim):
                z = self.z_dim - z - 1

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

                omega = trace(p_t[mask, z]) / (
                    trace(self.P_hat[t, mask]) + trace(p_t[mask, z])
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
        # CALCFilter.w_s = 2
        # CALCFilter.w_d = 1

        R = build_r_matrix(
            pos_error=1, pos_velo_error=0, d_pos_error=0, d_velo_error=0, dims=4
        ).to(self.device)
        H = build_h_matrix(dims=4).to(self.device)

        # # # # # scale R to a x,dim x dim matrix
        P_mod = (
            H.T @ R @ H
        )  # / 10 #+ CALCFilter.P_mod * torch.eye(self.x_dim).to(self.device)

        # self.P[:, ...]  += P_mod.clone() #+ CALCFilter.P_mod * torch.eye(self.x_dim).to(self.device)
        # self.P_hat[0, ...] = P_mod
        # # self.P = torch.clamp(self.P, min=0, max=15)
        # self.P_hat = torch.clamp(self.P_hat, min=0, max=1)
        self.P = (
            torch.clamp(
                self.P,
                min=1e-9,
            )
            # + P_mod
        )

        # where are there more than 1 measurement? When this happens, we augment the uncertainty
        # P_slicer = ((self.X.abs().sum(axis=-1) > 0).sum(axis=2) > 1)
        # self.P[P_slicer] += P_mod

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
            self.P_hat[t] = F @ self.P_hat[t - 1,] @ F.transpose(-1, -2)  + Q

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


class MeasurementCI(IMF):
    def apply_filter(self) -> None:
        H = build_h_matrix().to(self.device)
        R = build_r_matrix(pos_error=2.5, d_pos_error=1.5).to(self.device)
        eye = torch.eye(self.x_dim).to(self.device)
        # diag_mask = torch.eye(self.x_dim).byte()
        # other_mask = ~diag_mask

        for t in tqdm(range(1, self.t_dim)):
            x_t = self.X[t]
            # p_t = self.P[t]
            z_t = x_t @ H.T

            F = CALCFilter.F_static(
                dt_vect=self.Dt[t, :, 0],
                shape=(self.v_dim, self.x_dim, self.x_dim),
            ).to(self.device)

            # Q = CALCFilter.Q_static(
            #     dt_vect=self.Dt[t, :, 0],
            #     shape=(self.v_dim, self.x_dim, self.x_dim),
            # ).to(self.device)

            self.X_hat[t] = (F @ self.X_hat[t - 1,].unsqueeze(-1)).squeeze()
            self.P_hat[t] = F @ self.P_hat[t - 1,] @ F.transpose(-1, -2)  # + Q

            omega = torch.tensor(0.5).to(self.device)

            z_hat_t = self.X_hat[t] @ H.T

            for z in range(self.z_dim):
                mask = x_t[:, z].abs().sum(axis=-1) > 0.1

                y = z_t[mask, z] - z_hat_t[mask]

                sigma_y = H @ ((1 / omega) * self.P_hat[t, mask]) @ H.T + (
                    R / (1 - omega)
                )
                K = (1 / omega) * self.P_hat[t, mask] @ H.T @ torch.linalg.pinv(sigma_y)

                self.P_hat[t, mask] = (eye - K @ H) @ (
                    (1 / omega) * self.P_hat[t, mask]
                ) @ (eye - K @ H).mT + K @ (1 / (1 - omega) * R) @ K.mT

                self.X_hat[t, mask] = (
                    self.X_hat[t, mask] + (K @ y.unsqueeze(-1)).squeeze()
                ).squeeze()


# class SplitMeasurementCI(IMF):

#     def __init__(self, df: pl.DataFrame, gpu: bool = True, s_col: str = "s") -> None:
#         super().__init__(df, gpu, s_col)


#     def apply_filter(self) -> None:
#         H = build_h_matrix().to(self.device)
#         R = build_r_matrix(pos_error=3, d_pos_error=1).to(self.device)
#         R_d =

#         for t in tqdm(range(1, self.t_dim)):
#             x_t = self.X[t]
#             p_t = self.P[t]
#             z_t = x_t @ H.T

#             F = CALCFilter.F_static(
#                 dt_vect=self.Dt[t, :, 0],
#                 shape=(self.v_dim, self.x_dim, self.x_dim),
#             ).to(self.device)

#             Q = CALCFilter.Q_static(
#                 dt_vect=self.Dt[t, :, 0],
#                 shape=(self.v_dim, self.x_dim, self.x_dim),
#             ).to(self.device)

#             # split the Q matrix
#             Q_d = Q.clone()
#             Q_d.masked_fill(diag_mask, 0)
#             Q_i = Q.clone()
#             Q_i.masked_fill(other_mask, 0)

#             self.X_hat[t] = (F @ self.X_hat[t - 1,].unsqueeze(-1)).squeeze()

#             # split the P matrix
#             P_d = self.P_hat[t, :, ].clone()
#             P_d.masked_fill(diag_mask, 0)
#             P_i = self.P_hat[t, :, ].clone()
#             P_i.masked_fill(other_mask, 0)

#             omega = torch.tensor(0.5).to(self.device)

#             for z in range(self.z_dim):
#                 mask = x_t[:, z].abs().sum(axis=-1) > 0.1

#                 z_hat_t = self.X_hat[t] @ H.T
#                 y = z_t[mask, z] - z_hat_t[mask]

#                 sigma_y = H @ (
#                     (1 / omega) * self.P_hat[t, mask]
#                 ) @ H.T + (R / (1 - omega))
#                 K = (
#                     (1 / omega)
#                     * self.P_hat[t, mask]
#                     @ H.T
#                     @ torch.linalg.pinv(sigma_y)
#                 )

#                 self.P_hat[t, mask] = (eye - K @ H) @ (
#                     (1 / omega) * self.P_hat[t, mask]
#                 ) @ (eye - K @ H).mT + K @ (1 / (1 - omega) * R) @ K.mT

#                 self.X_hat[t, mask] = (
#                     self.X_hat[t, mask] + (K @ y.unsqueeze(-1)).squeeze()
#                 ).squeeze()


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
        for chunk_df in (
            filter_df
            # # .filter(pl.col("filter"))
            .with_columns(
                (pl.col("vehicle_id_int") // batch_size).alias("chunk")
            ).partition_by("chunk")
            # tqdm(
            #     # list(
            #     gen_chunks(
            #         filter_df,
            #         chunk_size=batch_size,
            #         vehicle_index="vehicle_id_int",
            #         time_index="time_index",
            #     )
            #     # )
            # )
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
        # create a list of the chunks
        # for chunk_df in (
        #         df
        #         # .filter(pl.col("filter"))
        #         .with_columns(
        #             (pl.col("vehicle_id_int") // batch_size).alias("chunk")
        #         ).partition_by("chunk")
        #         # tqdm(
        #         #     list(
        #         #         gen_chunks(
        #         #             df,
        #         #             chunk_size=batch_size,
        #         #             vehicle_index="vehicle_id_int",
        #         #             time_index="time_index",
        #         #         )
        #         #     )
        #         # )
        #     ):
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

    # CALCFilter.w_s = 1
    # CALCFilter.w_d = 2

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
