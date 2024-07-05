import gc
from itertools import permutations
import numpy as np
import torch
import polars as pl

from src.filters.vectorized_kalman import (
    pick_device,
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
        for ext, measure_list in zip(["_leader", ""], [Z_leaders, Z_followers]):
            measure_list.append(
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

    R = (
        build_r_matrix(d_pos_error=np.sqrt(1.5), pos_error=np.sqrt(5 / 3))
        .to(torch.float32)
        .to(device)
    )

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
    permute: bool = False,
    maha: bool = False,
) -> pl.DataFrame:
    device = pick_device(gpu)

    H = build_h_matrix(dims=dims).to(device)
    R = (
        build_r_matrix(d_pos_error=np.sqrt(1.5), pos_error=np.sqrt(5 / 3), dims=dims)
        .to(torch.float32)
        .to(device)
    )
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
    if maha:
        d = d.detach().cpu()
    else:
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
