from functools import lru_cache
from typing import List, Tuple
import torch
from torch import FloatTensor
import numpy as np
import polars as pl
import gc

# this is
# ruff: noqa: F821
from torchtyping import TensorType
from tqdm import tqdm


# Disable gradient calculation for speed
torch.set_grad_enabled(False)


def build_h_matrix() -> FloatTensor:
    """
    Build the H matrix for the Kalman filter.

    IDK why this is a function, but it is. ChatGPT made me do it
    """
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


def build_r_matrix(pos_error: float = 0.5, d_pos_error: float = 0.5) -> FloatTensor:
    """
    Build the R matrix for the Kalman filter

    This comes from https://www.smartmicro.com/fileadmin/media/Downloads/Traffic_Radar/Sensor_Data_Sheets__24_GHz_/UMRR-12_Type_48_TRUGRD_TM_Data_Sheet.pdf

    """
    discretization_error = (
        0.1  # constant uncertainty based on the discretization of the frenet frame
    )

    return torch.Tensor(
        np.array(
            [
                [pos_error + discretization_error, 0, 0, 0],
                [0, 0.5 + discretization_error, 0, 0],
                [0, 0, d_pos_error + discretization_error, 0],
                [0, 0, 0, 0.5 + discretization_error],
            ]
        )
        ** 2,
    )


def Q_white_noise(
    dt: FloatTensor,
    shape: Tuple[int, ...],
    ws: FloatTensor,
    wd: FloatTensor,
) -> FloatTensor:
    """
    Build the Q matrix for the Kalman filter. This is a white noise model
    """

    Q = torch.zeros(shape)

    ws_sq = ws**2
    wd_sq = wd**2

    # block build
    for offset, noise in [(0, ws_sq), (3, wd_sq)]:
        Q[..., 0 + offset, 0 + offset] = 1 / 4 * dt**4 * noise
        Q[..., 1 + offset, 0 + offset] = 1 / 2 * dt**3 * noise
        Q[..., 2 + offset, 0 + offset] = 1 / 2 * dt**2 * noise

        Q[..., 0 + offset, 1 + offset] = 1 / 2 * dt**3 * noise
        Q[..., 0 + offset, 2 + offset] = 1 / 2 * dt**2 * noise

        Q[..., 1 + offset, 1 + offset] = 1 * dt**2 * noise
        Q[..., 1 + offset, 2 + offset] = 1 * dt * noise

        Q[..., 2 + offset, 1 + offset] = dt * noise
        Q[..., 2 + offset, 2 + offset] = 1 * noise

    return Q


def pick_device(gpu: bool = True) -> torch.device:
    if not gpu:
        return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.has_mps:
        return torch.device("mps")


class _VectorizedKalmanFilter:
    x_dim = 6
    z_dim = 4

    P_mod = 1

    w_s = 8
    w_d = 1

    stop_speed_threshold = 0.5

    def __init__(
        self,
        df: pl.DataFrame = None,
        z: TensorType["time", "vehicle", "z_dim"] = None,
        dt: TensorType["time", "vehicle"] = None,
        predict_mask: TensorType["time", "vehicle"] = None,
        inds: TensorType = None,
        gpu: bool = True,
    ) -> None:
        if z is None:
            self._device = pick_device(gpu=gpu)
        else:
            self._device = z.device

        self.R: TensorType["z_dim", "z_dim"] = build_r_matrix().to(self._device)

        if inds is None:
            assert df is not None, "Must pass either df or the other arguments"
            # build the matrices
            self._inds: TensorType = torch.from_numpy(
                df[["time_ind", "vehicle_ind"]].to_numpy()
            ).to(self._device)
        else:
            self._inds: TensorType = inds.to(self._device)

        # make sure the vehicle indices start at 0 (for batching)
        self._inds[:, 1] -= self._inds[:, 1].min()

        self.t_dim: int = (self._inds[:, 0].max() + 1).cpu()
        self.v_dim: int = (self._inds[:, 1].max() + 1).cpu()

        # store H matrix
        self._H: TensorType["x_dim", "z_dim"] = build_h_matrix().to(self._device)
        self._I: TensorType["x_dim", "x_dim"] = torch.eye(
            self.x_dim, device=self._device
        )

        # store the measurements
        if z is None:
            self._z: TensorType["time", "vehicle", "z_dim"] = torch.zeros(
                (self.t_dim, self.v_dim, self.z_dim), device=self._device
            )
            self._z[self._inds[:, 0], self._inds[:, 1]] = torch.Tensor(
                df["z"].to_numpy(writable=True)
            ).to(self._device)
        else:
            # this is for IMM filter, where z can be shared across filters
            self._z: TensorType["time", "vehicle", "z_dim"] = z.to(self._device)

        if predict_mask is None:
            # save where the all 0 as predictions are
            self._predict_mask: TensorType[
                "time",
                "vehicle",
            ] = torch.zeros(
                (self.t_dim, self.v_dim),
                dtype=torch.bool,
                device=self._device,
            )
            self._predict_mask[self._inds[:, 0], self._inds[:, 1]] = torch.BoolTensor(
                ~df["missing_data"]
                .to_numpy(
                    writable=True,
                )
                .astype(np.bool_),
            ).to(self._device)
        else:
            # this is for IMM filter, where predict_mask can be shared across filters
            self._predict_mask: TensorType[
                "time", "vehicle", "x_dim"
            ] = predict_mask.to(self._device)

        # save the time differences
        if dt is None:
            self._dt: TensorType["time", "vehicle"] = torch.zeros(
                (self.t_dim, self.v_dim), device=self._device
            )
            self._dt[self._inds[:, 0], self._inds[:, 1]] = torch.Tensor(
                df["time_diff"].to_numpy().copy(),
            ).to(self._device)
        else:
            # this is for IMM filter, where dt can be shared across filters
            self._dt: TensorType["time", "vehicle", "x_dim"] = dt.to(self._device)

        # initialize filt_x and z as dense tensors
        self._x: TensorType["time", "vehicle", "x_dim"] = torch.zeros(
            (self.t_dim, self.v_dim, self.x_dim), device=self._device
        )
        self._x[0, :, :] = self._z[0] @ self._H

        # create P matrix
        self._P: TensorType["time", "vehicle", "x_dim", "x_dim"] = torch.zeros(
            (self.t_dim, self.v_dim, self.x_dim, self.x_dim), device=self._device
        )
        self._P[self._inds[:, 0], self._inds[:, 1]] = (
            (torch.eye(self.x_dim) * self.P_mod)
            .repeat(len(self._inds), 1, 1)
            .to(self._device)
        )

        self._F: TensorType["vehicle", "x_dim", "x_dim"] = None
        self._Q: TensorType["vehicle", "x_dim", "x_dim"] = None

        self._S: TensorType["vehicle", "x_dim", "x_dim"] = None
        self._y: TensorType["vehicle", "x_dim"] = None

        self._log2pi: TensorType = torch.log(
            torch.tensor(2 * np.pi, device=self._device)
        )

    @classmethod
    def from_other_filter(
        cls,
        other_filter: "_VectorizedKalmanFilter",
    ) -> "_VectorizedKalmanFilter":
        return cls(
            z=other_filter.z,
            dt=other_filter.dt,
            predict_mask=other_filter.predict_mask,
            inds=other_filter.inds,
        )

    @classmethod
    def from_prebuilt_tensors(
        cls,
        z: TensorType["time", "vehicle", "z_dim"],
        dt: TensorType["time", "vehicle"],
        predict_mask: TensorType["time", "vehicle"],
        inds: TensorType,
    ) -> "_VectorizedKalmanFilter":
        return cls(
            z=z,
            dt=dt,
            predict_mask=predict_mask,
            inds=inds,
        )

    def F(self, t_ind: int) -> TensorType["vehicle", "x_dim", "x_dim"]:
        # this should be overridden by the child class
        raise NotImplementedError

    @classmethod
    def Q_static(
        cls,
        dt_vect: int,
        shape: Tuple[int, ...],
        speed_info: TensorType["vehicle"] = None,
        mask: TensorType["vehicle"] = None,
    ) -> TensorType["vehicle", "x_dim", "x_dim"]:
        # this should be overridden by the child class
        raise NotImplementedError

    def Q(self, t_ind: int) -> TensorType["vehicle", "x_dim", "x_dim"]:
        # this should be overridden by the child class
        return self.Q_static(
            self._dt[t_ind],
            (self.v_dim, self.x_dim, self.x_dim),
            self._z[t_ind, :, 0],  # speed info
            self._predict_mask[t_ind],
        ).to(self.device)

    @property
    def inds(
        self,
    ) -> TensorType:
        return self._inds

    @property
    def device(
        self,
    ) -> torch.device:
        return self._device

    @property
    def x(
        self,
    ) -> TensorType["time", "vehicle", "x_dim"]:
        return self._x

    @property
    def z(
        self,
    ) -> TensorType["time", "vehicle", "z_dim"]:
        return self._z

    @property
    def dt(
        self,
    ) -> TensorType["time", "vehicle"]:
        return self._dt

    @property
    def P(
        self,
    ) -> TensorType["time", "vehicle", "x_dim", "x_dim"]:
        return self._P

    @property
    def predict_mask(
        self,
    ) -> TensorType["time", "vehicle"]:
        return self._predict_mask

    @classmethod
    def stop_adjusted_w_s(
        cls,
        speed_info: TensorType["vehicle"],
    ) -> TensorType["vehicle"]:
        return torch.where(
            speed_info.abs() > cls.stop_speed_threshold,
            cls.w_s,
            0,
        )

    def predict(
        self,
        t_ind: int,
        x: TensorType["vehicle", "x_dim"] = None,
        P: TensorType["vehicle", "x_dim", "x_dim"] = None,
    ) -> None:
        # basic Kalman Filter prediction step
        self._F = self.F(t_ind)
        self._Q = self.Q(t_ind)

        last_x = self._x[t_ind - 1] if x is None else x
        last_P = self._P[t_ind - 1] if P is None else P

        if t_ind > 0:
            self._x[t_ind, :, :] = (self._F @ last_x.unsqueeze(-1)).squeeze()
            self._P[t_ind, :, :] = (
                self._F @ last_P @ self._F.transpose(-1, -2) + self._Q
            )

    def update(self, t_ind: int) -> None:
        # Linear Kalman Filter update step
        if t_ind >= self.t_dim:
            return
        local_mask = self._predict_mask[t_ind]
        y = (
            self._z[t_ind, local_mask]
            - (self._H @ self._x[t_ind, local_mask].unsqueeze(-1)).squeeze()
        )

        PHT = self._P[t_ind, local_mask] @ self._H.T
        S = self._H @ PHT + self.R
        SI = torch.inverse(S)
        K = PHT @ SI
        I_KH = self._I - K @ self._H

        self._x[t_ind, local_mask, :] = (
            self._x[t_ind, local_mask] + torch.matmul(K, y.unsqueeze(-1)).squeeze()
        )

        # Joseph form
        self._P[t_ind, local_mask, :, :] = torch.matmul(
            I_KH @ self._P[t_ind, local_mask],
            I_KH.transpose(1, 2),
        ) + torch.matmul(K, self.R) @ K.transpose(1, 2)

        self._update_t = t_ind
        self._S = S.clone()
        self._y = y.clone()

    def apply_filter(self, detach: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        # iterate over time
        for t in tqdm(range(self._x.shape[0])):
            self.predict(t)
            self.update(t)
        if detach:
            return (
                self._x.to("cpu").detach().numpy(),
                self._P.to("cpu").detach().numpy(),
            )
        else:
            return self._x, self._P

    def log_likelihood(
        self,
        t_ind: int,
    ) -> FloatTensor:
        # compute the log likelihood of the measurement
        assert t_ind == self._update_t, "Must call update before log_likelihood"
        # mask = self._predict_mask[t_ind]

        # do it in a numerically stable way
        # https://gregorygundersen.com/blog/2019/10/30/scipy-multivariate/
        i = 0
        mask = torch.ones(self._S.shape[0], dtype=torch.bool, device=self._device)
        likelihood = torch.zeros_like(mask, dtype=torch.float32, device=self._device)
        while i < 10:
            jitter = 1e-6 * i * torch.eye(self._S[mask].shape[-1], device=self._device)
            try:
                vals, vecs = torch.linalg.eigh(self._S[mask] + jitter)
                logdet = torch.log(vals).sum(
                    axis=-1,
                )
                valsinv = 1 / vals
                U = vecs * valsinv.sqrt().unsqueeze(-1)
                rank = torch.linalg.matrix_rank(U)
                maha = ((U @ self._y[mask].unsqueeze(-1)).squeeze() ** 2).sum(axis=-1)
                likelihood[mask] = -0.5 * (self._log2pi * rank + logdet + maha)
                break
            except RuntimeError as e:
                # find which vehicle is causing the problem
                bad_inds = torch.where(
                    torch.any(
                        torch.isnan(
                            U,
                        ),
                        -2,
                    ).any(-1)
                )[0]

                # bad_veh = bad_inds[0]
                # # the bad vehicle is the
                # bad_veh_ind = torch.where(
                #     bad_veh == (self._predict_mask[t_ind].cumsum(-1) - 1)
                # )[0][0]
                # print(f"bad vehicle: {bad_veh_ind}")
                # print(f"bad filter: {self.__class__.__name__}")

                # mask out the bad vehicle (return a large negative number so it doesn't get picked)
                mask[bad_inds] = False
                likelihood[bad_inds] = -1 * torch.finfo(likelihood.dtype).max

                # print(f"jitter: {jitter}")

        return likelihood

    def cleanup(
        self,
    ) -> None:
        # iterate through variables in the class and del if it is a tensor
        for attr in dir(self):
            if isinstance(getattr(self, attr), torch.Tensor):
                try:
                    setattr(self, attr, None)
                except AttributeError:
                    pass

        # clear the cache
        torch.cuda.empty_cache()

    @classmethod
    def rts_smoother(
        cls,
        Xs: TensorType["time", "vehicle", "x_dim"],
        Ps: TensorType["time", "vehicle", "x_dim", "x_dim"],
        dts: TensorType["time", "vehicle"],
        device: torch.device = None,
    ) -> Tuple[
        TensorType["time", "vehicle", "x_dim"],
        TensorType["time", "vehicle", "x_dim", "x_dim"],
    ]:
        if len(Xs) != len(Ps):
            raise ValueError("length of Xs and Ps must be the same")

        n, v_dim, dim_x = Xs.shape

        if device is None:
            device = Xs.device

        # smoother gain
        out_Xs = Xs.unsqueeze(-1).clone()
        for k in tqdm(range(n - 2, -1, -1)):
            mask = dts[k + 1] > 0

            F_last = cls.F_static(dts[k + 1, mask], (mask.sum(), dim_x, dim_x)).to(
                device
            )
            Q_last = cls.Q_static(
                dts[k + 1, mask], (mask.sum(), dim_x, dim_x), Xs[k + 1, mask, 0]
            ).to(device)

            Pp = F_last @ Ps[k, mask] @ F_last.transpose(-1, -2) + Q_last

            # replace pinv with torch.linalg.lstsq(A, B).solution
            # K = torch.linalg.lstsq(Ps[k, mask] @ F_last.transpose(-1, -2), torch.eye(dim_x, device=device)).solution

            K = (
                Ps[k, mask]
                @ F_last.transpose(-1, -2)
                @ torch.linalg.pinv(Pp, hermitian=True)
            )

            out_Xs[k, mask] += K @ (out_Xs[k + 1, mask] - F_last @ out_Xs[k, mask])
            Ps[k, mask] += K @ (Ps[k + 1, mask] - Pp) @ K.transpose(-1, -2)

        return out_Xs.squeeze(), Ps


class CALKFilter(_VectorizedKalmanFilter):
    # w_s = 10
    w_d = 5

    def __init__(
        self,
        df: pl.DataFrame = None,
        z: TensorType = None,
        dt: TensorType = None,
        predict_mask: TensorType = None,
        inds: TensorType = None,
        **kwargs,
    ) -> None:
        super().__init__(df, z, dt, predict_mask, inds, **kwargs)

    @staticmethod
    def F_static(
        dt_vect: int, shape: Tuple[int, ...]
    ) -> TensorType["vehicle", "x_dim", "x_dim"]:
        F = torch.zeros(shape)
        F[..., 0, 0] = 1
        F[..., 0, 1] = dt_vect
        F[..., 0, 2] = 0.5 * dt_vect**2

        F[..., 1, 1] = 1
        F[..., 1, 2] = dt_vect

        F[..., 2, 2] = 1

        F[..., 3, 3] = 1
        return F

    def F(self, t_ind: int) -> TensorType["vehicle", "x_dim", "x_dim"]:
        return self.F_static(
            self._dt[t_ind],
            (self.v_dim, self.x_dim, self.x_dim),
        ).to(self._device)

    @classmethod
    def Q_static(
        cls,
        dt_vect: int,
        shape: Tuple[int, ...],
        speed_info: TensorType["vehicle"] = None,
        mask: TensorType["vehicle"] = None,
    ) -> TensorType["vehicle", "x_dim", "x_dim"]:
        w_s = cls.stop_adjusted_w_s(speed_info) if speed_info is not None else cls.w_s

        # replace predictions with noise
        if (mask is not None) and (speed_info is not None):
            w_s = torch.where(
                mask,
                w_s,
                0,
            )

        Q = Q_white_noise(
            dt_vect,
            shape,
            w_s,
            cls.w_d,
        )

        # zero out the d_dot and dd_dot terms
        Q[..., 4, :] = 0
        Q[..., 5, :] = 0
        Q[..., :, 4] = 0
        Q[..., :, 5] = 0

        return Q

    @classmethod
    def rts_smoother(
        cls,
        Xs: TensorType["time", "vehicle", "x_dim"],
        Ps: TensorType["time", "vehicle", "x_dim", "x_dim"],
        dts: TensorType["time", "vehicle"],
        device: torch.device = None,
    ) -> Tuple[
        TensorType["time", "vehicle", "x_dim"],
        TensorType["time", "vehicle", "x_dim", "x_dim"],
    ]:
        if len(Xs) != len(Ps):
            raise ValueError("length of Xs and Ps must be the same")

        n, v_dim, dim_x = Xs.shape

        if device is None:
            device = Xs.device

        # smoother gain
        out_Xs = Xs.unsqueeze(-1).clone()
        for k in tqdm(range(n - 2, -1, -1)):
            mask = dts[k + 1] > 0

            F_last = cls.F_static(dts[k + 1, mask], (mask.sum(), dim_x, dim_x)).to(
                device
            )
            Q_last = cls.Q_static(
                dts[k + 1, mask], (mask.sum(), dim_x, dim_x), Xs[k + 1, mask, 0]
            ).to(device)

            Pp = F_last @ Ps[k, mask] @ F_last.transpose(-1, -2) + Q_last

            K = Ps[k, mask] @ F_last.transpose(-1, -2) @ torch.linalg.pinv(Pp)

            out_Xs[k, mask] += K @ (out_Xs[k + 1, mask] - F_last @ out_Xs[k, mask])
            Ps[k, mask] += K @ (Ps[k + 1, mask] - Pp) @ K.transpose(-1, -2)

        return out_Xs.squeeze(), Ps


class CVLKFilter(_VectorizedKalmanFilter):
    # w_s = 10
    w_d = 1

    def __init__(
        self,
        df: pl.DataFrame = None,
        z: TensorType = None,
        dt: TensorType = None,
        predict_mask: TensorType = None,
        inds: TensorType = None,
        **kwargs,
    ) -> None:
        super().__init__(df, z, dt, predict_mask, inds, **kwargs)

    def F(self, t_ind: int) -> TensorType["vehicle", "x_dim", "x_dim"]:
        return self.F_static(
            self._dt[t_ind],
            (self.v_dim, self.x_dim, self.x_dim),
        ).to(self._device)

    @staticmethod
    def F_static(
        dt_vect: int, shape: Tuple[int, ...]
    ) -> TensorType["vehicle", "x_dim", "x_dim"]:
        F = torch.zeros(shape)

        F[:, 0, 0] = 1
        F[:, 0, 1] = dt_vect
        F[:, 1, 1] = 1
        F[:, 3, 3] = 1
        return F

    @classmethod
    def Q_static(
        cls,
        dt_vect: int,
        shape: Tuple[int, ...],
        speed_info: TensorType["vehicle"] = None,
        mask: TensorType["vehicle"] = None,
    ) -> TensorType["vehicle", "x_dim", "x_dim"]:
        w_s = cls.stop_adjusted_w_s(speed_info) if speed_info is not None else cls.w_s
        # replace predictions with noise
        if (mask is not None) and (speed_info is not None):
            w_s = torch.where(
                mask,
                w_s,
                0,
            )
        Q = Q_white_noise(
            dt_vect,
            shape,
            w_s,
            cls.w_d,
        )

        # zero out the d_dot and dd_dot terms
        Q[..., 2, :] = 0  # no acceleration
        Q[..., :, 2] = 0  # no acceleration

        Q[..., 4, :] = 0
        Q[..., 5, :] = 0
        Q[..., :, 4] = 0
        Q[..., :, 5] = 0
        return Q @ Q.transpose(-1, -2)


class CALCFilter(_VectorizedKalmanFilter):
    # w_s = 10
    w_d = 1

    def __init__(
        self,
        df: pl.DataFrame = None,
        z: TensorType = None,
        dt: TensorType = None,
        predict_mask: TensorType = None,
        inds: TensorType = None,
        **kwargs,
    ) -> None:
        super().__init__(df, z, dt, predict_mask, inds, **kwargs)

    @staticmethod
    def F_static(
        dt_vect: int, shape: Tuple[int, ...]
    ) -> TensorType["vehicle", "x_dim", "x_dim"]:
        F = torch.zeros(shape)

        F[..., 0, 0] = 1
        F[..., 0, 1] = dt_vect
        F[..., 0, 2] = 0.5 * dt_vect**2

        F[..., 1, 1] = 1
        F[..., 1, 2] = dt_vect

        F[..., 2, 2] = 1

        F[..., 3, 3] = 1
        F[..., 3, 4] = dt_vect
        F[..., 3, 5] = 0.5 * dt_vect**2

        F[..., 4, 4] = 1
        F[..., 4, 5] = dt_vect

        F[..., 5, 5] = 1

        return F

    def F(self, t_ind: int) -> TensorType["vehicle", "x_dim", "x_dim"]:
        return self.F_static(
            self._dt[t_ind],
            (self.v_dim, self.x_dim, self.x_dim),
        ).to(self._device)

    @classmethod
    def Q_static(
        cls,
        dt_vect: int,
        shape: Tuple[int, ...],
        speed_info: TensorType["vehicle"] = None,
        mask: TensorType["vehicle"] = None,
    ) -> TensorType["vehicle", "x_dim", "x_dim"]:
        w_s = cls.stop_adjusted_w_s(speed_info) if speed_info is not None else cls.w_s

        # replace predictions with noise
        if (mask is not None) and (speed_info is not None):
            w_s = torch.where(
                mask,
                w_s,
                0,
            )

        Q = Q_white_noise(
            dt_vect,
            shape,
            w_s,
            cls.w_d,
        )

        return Q


class IMMFilter:
    def __init__(
        self,
        df: pl.DataFrame,
        filters: Tuple[str] = ("CALC", "CALK", "CVLK"),
        M: np.ndarray = None,
        mu: np.ndarray = None,
        gpu: bool = True,
    ) -> None:
        self.f_dim = len(filters)

        # initalize the first filter (this is where the z, dt, and predict_mask are stored)
        self._filters: List[_VectorizedKalmanFilter] = [
            eval(f"{filters[0].upper()}Filter")(df, gpu=gpu)
        ]

        # get the dimensions
        self.t_dim = self._filters[0].t_dim
        self.v_dim = self._filters[0].v_dim
        self.x_dim = self._filters[0].x_dim

        # initialize the other filters
        self._filters.extend(
            eval(f"{f.upper()}Filter").from_other_filter(
                self._filters[0],
            )
            for f in filters[1:]
        )

        self._device: torch.device = self._filters[0].device

        if mu is None:
            mu = np.ones(self.f_dim) / self.f_dim
        assert (
            len(mu) == self.f_dim
        ), "mu must have length equal to the number of filters"
        # initialize the mixing matrix
        self._mu: TensorType["time", "vehicle", "filter"] = torch.zeros(
            (self._filters[0].t_dim, self._filters[0].v_dim, self.f_dim),
            device=self._device,
            dtype=torch.float32,
        )
        self._mu[0, :, :] = torch.from_numpy(mu.astype(np.float32)).to(self._device)

        # initialize the mixing probabilities
        if M is None:
            M = np.array(
                [
                    [0.9, 0.05, 0.05],
                    [0.05, 0.9, 0.05],
                    [0.05, 0.05, 0.9],
                ]
            )
        assert M.shape == (
            self.f_dim,
            self.f_dim,
        ), f"M must be a square matrix with shape ({self.f_dim}, {self.f_dim})"
        self.M = torch.from_numpy(M.astype(np.float32)).to(self._device)

        self._omega: TensorType["time", "vehicle", "filter", "filter"] = torch.zeros(
            (self._filters[0].t_dim, self._filters[0].v_dim, self.f_dim, self.f_dim),
            device=self._device,
            dtype=torch.float32,
        )

        # initalize cbar
        self._cbar: TensorType["vehicle", "filter"] = self.cbar(0)
        self._compute_mixing_probabilities(0)

        # initalize x & p
        self._x = torch.zeros(
            (self.t_dim, self.v_dim, self.x_dim),
            device=self._device,
            dtype=torch.float32,
        )

        self._P = torch.zeros(
            (self.t_dim, self.v_dim, self.x_dim, self.x_dim),
            device=self._device,
            dtype=torch.float32,
        )

        # add a prediction mask.
        # This is different than the filter's predict mask, as it indicates that lane-change predictions are not allowed
        self._predict_mask = torch.zeros(
            (
                self.t_dim,
                self.v_dim,
            ),
            device=self._device,
            dtype=torch.bool,
        )

        self._predict_mask[
            self._filters[0].inds[:, 0], self._filters[0].inds[:, 1]
        ] = torch.BoolTensor(
            df["prediction"].to_numpy(writable=True).astype(np.bool_),
        ).to(self._device)

        self._compute_state_estimate(0)

    @lru_cache(maxsize=1)
    def cbar(self, t_ind: int) -> TensorType["vehicle", "filter"]:
        return self._mu[t_ind] @ self.M

    def filter_x(self, t_ind: int) -> TensorType["vehicle", "filter", "x_dim"]:
        return torch.stack([f.x[t_ind] for f in self._filters], dim=1)

    def filter_P(self, t_ind: int) -> TensorType["vehicle", "filter", "x_dim", "x_dim"]:
        return torch.stack([f.P[t_ind] for f in self._filters], dim=1)

    def filter_H(
        self,
    ) -> TensorType["filter", "z_dim", "x_dim"]:
        return torch.stack([f._H for f in self._filters], dim=0)

    def filter_R(
        self,
    ) -> TensorType["filter", "z_dim", "z_dim"]:
        return torch.stack([f.R for f in self._filters], dim=0)

    def filter_I(
        self,
    ) -> TensorType["x_dim", "x_dim"]:
        # I need to stop accessing private variables
        return self._filters[0]._I

    def _compute_mixing_probabilities(self, t_ind: int) -> None:
        # doing again bc I don't want to store the mixing probabilities
        self._cbar = self.cbar(t_ind)
        # replace 0 or less than float min with float min to avoid underflow
        self._cbar[self._cbar <= torch.finfo(self._cbar.dtype).min] = torch.finfo().min

        omega = self.M * self._mu[t_ind, :, None, :] / self._cbar[:, :, None]
        # replace 0 or less than float min with float min to avoid underflow
        # omega has 2 dimensions, so need to do it in the last two dimensions
        omega[omega <= torch.finfo(omega.dtype).min] = torch.finfo().min

        self._omega[t_ind] = omega / omega.sum(dim=-1, keepdim=True)

    def _compute_state_estimate(self, t_ind: int) -> None:
        # compute the mixed state estimate
        all_x = self.filter_x(t_ind)
        all_P = self.filter_P(t_ind)

        self._x[t_ind] = torch.einsum(
            "abc,ab->ac",
            all_x,
            self._mu[t_ind],
        )
        # compute the mixed covariance estimate
        y = all_x - self._x[t_ind, :, None, :]
        outer = y[:, :, :, None] @ y[:, :, None, :]
        # create P matrix
        self._P[t_ind] = torch.einsum(
            "ab,abij->aij",
            self._mu[t_ind],
            outer + all_P,
        )

    def predict(self, t_ind: int) -> None:
        self._predict_vectorized(t_ind)
        self._compute_state_estimate(t_ind)

    def _predict_vectorized(self, t_ind: int) -> None:
        # do it in a completely vectorized way
        t_ind_l = max(t_ind - 1, 0)

        F = torch.stack(
            [f.F(t_ind) for f in self._filters],
            axis=1,
        )
        Q = torch.stack(
            [f.Q(t_ind) for f in self._filters],
            axis=1,
        )

        # mix the x and P matrices
        all_x = self.filter_x(t_ind_l)
        all_P = self.filter_P(t_ind_l)

        # transpose the omega matrix
        omega_me_crazy = self._omega[t_ind_l].transpose(-1, -2)

        # make the mixed x matrix
        xs = (omega_me_crazy[:, :, :, None] * all_x[:, :, None, :]).sum(axis=1)
        # make the mixed P matrix (a lot of broadcasting magic here)
        y = all_x[:, :, None, :] - xs[:, None, :, :]
        outer = y[:, :, :, :, None] @ y[:, :, :, None, :]
        Ps = (
            omega_me_crazy[:, :, :, None, None] * (outer + all_P[:, :, None, :, :])
        ).sum(axis=1)

        # predict at once
        if t_ind > 0:
            x_pred = (F @ xs.unsqueeze(-1)).squeeze(-1)
            P_pred = (F @ Ps) @ F.transpose(-1, -2) + Q
        else:
            x_pred = xs
            P_pred = Ps
        # update the x and P matrices for each filter
        for i in range(self.f_dim):
            self._filters[i].x[t_ind] = x_pred[:, i].clone()
            self._filters[i].P[t_ind] = P_pred[:, i].clone()

        return x_pred, P_pred

    def update(self, t_ind: int) -> None:
        mask = self._filters[0].predict_mask[t_ind]
        likelihoods = self.log_likelihood_vectorized(
            *self.update_vectorized(t_ind)
        ).exp()

        likelihoods[
            likelihoods <= torch.finfo(likelihoods.dtype).min
        ] = torch.finfo().min

        # update the mode probabilities. This is the spot! where they get updated
        # so have to use the last time step
        self._mu[t_ind, mask] = likelihoods * self.cbar(max(t_ind - 1, 0))[mask]
        self._mu[t_ind, ~mask] = self._mu[max(t_ind - 1, 0), ~mask]

        self._mu[t_ind, self._predict_mask[t_ind], 0] = 0
        self._mu[t_ind,] = self._mu[t_ind,] / self._mu[t_ind,].sum(dim=1, keepdim=True)

        # compute the mixing probabilities
        self._compute_mixing_probabilities(t_ind)

        # compute the state estimate
        self._compute_state_estimate(t_ind)

    def update_vectorized(
        self, t_ind: int
    ) -> Tuple[
        TensorType["filt", "masked_vehicle", "x_dim"],
        TensorType["filt", "masked_vehicle", "x_dim", "x_dim"],
    ]:
        local_mask = self._filters[0].predict_mask[t_ind]

        # get a z (shared across filters)
        z = self._filters[0].z[t_ind, local_mask]
        x = self.filter_x(t_ind)[local_mask]
        P = self.filter_P(t_ind)[local_mask]
        H = self.filter_H()
        R = self.filter_R()
        I = self.filter_I()  # noqa: E741

        y = z[:, None, :] - (H @ x.unsqueeze(-1)).squeeze()

        PHT = P @ H.transpose(-1, -2)
        S = H @ PHT + R
        try:
            SI = torch.linalg.pinv(
                S,
                hermitian=True,
            )
        except RuntimeError as e:
            print(f"pinverse failed at time {t_ind}")
            raise e
        K = PHT @ SI
        I_KH = I - K @ H

        x = x + torch.matmul(K, y.unsqueeze(-1)).squeeze()

        P = torch.matmul(
            I_KH @ P,
            I_KH.transpose(-1, -2),
        ) + torch.matmul(K, R) @ K.transpose(-1, -2)

        for i, f in enumerate(self._filters):
            f.x[t_ind, local_mask] = x[:, i].clone()
            f.P[t_ind, local_mask] = P[:, i].clone()

        return y, S

    def log_likelihood_vectorized(
        self,
        y: TensorType["filt", "masked_vehicle", "x_dim"],
        S: TensorType["filt", "masked_vehicle", "x_dim", "x_dim"],
    ) -> TensorType["filt", "masked_vehicle"]:
        # https://gregorygundersen.com/blog/2019/10/30/scipy-multivariate/

        try:
            vals, vecs = torch.linalg.eigh(S)
        except RuntimeError as e:
            print(f"eigh failed")
            raise e

        logdet = torch.log(vals).sum(
            axis=-1,
        )
        valsinv = 1 / vals
        U = vecs * valsinv.sqrt().unsqueeze(-1)
        rank = torch.linalg.matrix_rank(U)
        maha = ((U @ y.unsqueeze(-1)).squeeze() ** 2).sum(axis=-1)
        return -0.5 * (self._filters[0]._log2pi * rank + logdet + maha)

    def apply_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        # iterate over time
        try:
            for t in tqdm(range(self.t_dim)):
                self.predict(t)
                self.update(t)
        except torch.linalg.LinAlgError as e:
            print(f"LinAlgError at time {t}")
            raise e
        except Exception as e:
            print(f"Exception at time {t}")
            raise e

        # raise the error again
        return (
            self._x.to("cpu").numpy(),
            self._P.to("cpu").numpy(),
        )

    def cleanup(
        self,
    ) -> None:
        # iterate through variables in the class and del if it is a tensor
        for attr in dir(self):
            if isinstance(getattr(self, attr), torch.Tensor):
                # if it is a class attribute, delete it
                try:
                    delattr(self, attr)
                except AttributeError:
                    pass

        # cleanup all of the filters
        for f in self._filters:
            f.cleanup()

        self._filters = []
        gc.collect()
        # clear the cache
        torch.cuda.empty_cache()


def batch_imm_df(
    df: pl.DataFrame,
    filters: List[str],
    M: np.ndarray,
    mu: np.ndarray,
    gpu: bool = True,
    chunk_size: int = 10_000,
) -> pl.DataFrame:
    import pyarrow as pa

    filts = []

    # chunk the filters
    for chunk_df in tqdm(
        df.with_columns(
            (pl.col("vehicle_ind") // chunk_size).alias("chunk")
        ).partition_by("chunk")
    ):
        offset = chunk_df["vehicle_ind"].min()

        imm = IMMFilter(
            chunk_df.with_columns(
                (pl.col("vehicle_ind") - offset).alias("vehicle_ind")
            ),
            filters=filters,
            M=M,
            mu=mu,
            gpu=gpu,
        )

        x_filt, p_filt = imm.apply_filter()
        mu_filt = imm._mu.cpu().numpy()

        t_index = np.repeat(np.arange(imm.t_dim), imm.v_dim)
        # v_index = np.tile(np.arange(imm.v_dim) + , imm.t_dim)
        v_index = np.tile(np.arange(imm.v_dim), imm.t_dim) + offset
        x_filt = x_filt.reshape(-1, imm.x_dim)
        p_filt = p_filt.reshape(-1, imm.x_dim, imm.x_dim)
        p_calc = (
            tuple(filter(lambda f: isinstance(f, CALCFilter), imm._filters))[0]
            .P
            # .detach()
            .cpu()
            .numpy()
            .reshape(-1, imm.x_dim, imm.x_dim)
        )
        mu_filt = mu_filt.reshape(-1, imm.f_dim)
        filts.append(
            pl.DataFrame(
                {
                    "time_ind": t_index,
                    "vehicle_ind": v_index,
                    "s": x_filt[:, 0],
                    "s_velocity": x_filt[:, 1],
                    "s_accel": x_filt[:, 2],
                    "d": x_filt[:, 3],
                    "d_velocity": x_filt[:, 4],
                    "d_accel": x_filt[:, 5],
                    "P": pa.FixedSizeListArray.from_arrays(
                        p_filt.reshape(-1), imm.x_dim * imm.x_dim
                    ),
                    "P_CALC": pa.FixedSizeListArray.from_arrays(
                        p_calc.reshape(-1), imm.x_dim * imm.x_dim
                    ),
                    **{
                        f"mu_{f}": mu_filt[:, i]
                        for i, f in enumerate(filt.upper() for filt in filters)
                    },
                }
            ).with_columns(
                [
                    pl.col("time_ind").cast(pl.Int32),
                    pl.col("vehicle_ind").cast(pl.Int32),
                ]
            )
        )

        imm.cleanup()
        del imm
        gc.collect()
        torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()
    return pl.concat(filts)



def batch_kalman_df(
    df: pl.DataFrame,
    filter: str,
    gpu: bool = True,
    chunk_size: int = 10_000,
) -> pl.DataFrame:
    import pyarrow as pa

    filts = []

    # chunk the filters
    for chunk_df in tqdm(
        df.with_columns(
            (pl.col("vehicle_ind") // chunk_size).alias("chunk")
        ).partition_by("chunk")
    ):
        offset = chunk_df["vehicle_ind"].min()

        filter: _VectorizedKalmanFilter = eval(f"{filter}")

        kf = filter(
            chunk_df.with_columns(
                (pl.col("vehicle_ind") - offset).alias("vehicle_ind")
            ),
            gpu=gpu,
        )
        

        x_filt, p_filt = kf.apply_filter()
        # mu_filt = imm._mu.cpu().numpy()

        t_index = np.repeat(np.arange(kf.t_dim), kf.v_dim)
        # v_index = np.tile(np.arange(imm.v_dim) + , imm.t_dim)
        v_index = np.tile(np.arange(kf.v_dim), kf.t_dim) + offset
        x_filt = x_filt.reshape(-1, kf.x_dim)
        p_filt = p_filt.reshape(-1, kf.x_dim, kf.x_dim)
        # p_calc = (
        #     tuple(filter(lambda f: isinstance(f, CALCFilter), kf._filters))[0]
        #     .P
        #     # .detach()
        #     .cpu()
        #     .numpy()
        #     .reshape(-1, imm.x_dim, imm.x_dim)
        # )
        # mu_filt = mu_filt.reshape(-1, imm.f_dim)
        filts.append(
            pl.DataFrame(
                {
                    "time_ind": t_index,
                    "vehicle_ind": v_index,
                    "s": x_filt[:, 0],
                    "s_velocity": x_filt[:, 1],
                    "s_accel": x_filt[:, 2],
                    "d": x_filt[:, 3],
                    "d_velocity": x_filt[:, 4],
                    "d_accel": x_filt[:, 5],
                    "P": pa.FixedSizeListArray.from_arrays(
                        p_filt.reshape(-1), kf.x_dim * kf.x_dim
                    ),
                    # "P_CALC": pa.FixedSizeListArray.from_arrays(
                    #     p_calc.reshape(-1), imm.x_dim * imm.x_dim
                    # ),
                    # **{
                    #     f"mu_{f}": mu_filt[:, i]
                    #     for i, f in enumerate(filt.upper() for filt in filters)
                    # },
                }
            ).with_columns(
                [
                    pl.col("time_ind").cast(pl.Int32),
                    pl.col("vehicle_ind").cast(pl.Int32),
                ]
            )
        )

        kf.cleanup()
        del kf
        gc.collect()
        torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()
    return pl.concat(filts)
