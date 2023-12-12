from typing import List
import numpy as np
import torch
import polars as pl
from pomegranate.distributions import Normal, StudentT
from pomegranate.gmm import GeneralMixtureModel


def build_lane_model(
    lane_centers: List[float],
    train_data: pl.Series,
    priors: List[float] = None,
    variance: float = 1,
) -> GeneralMixtureModel:
    models = []
    for lane_center in lane_centers:
        models.append(
            StudentT(
                dofs=len(train_data) - 1,
                means=[
                    lane_center,
                ],
                covs=[
                    variance,
                ] if variance is not None else None,
                covariance_type="diag",
            )
        )

    return GeneralMixtureModel(
        distributions=models,
        # priors=priors,
        tol=1e-6,
        verbose=True
    ).fit(torch.from_numpy(train_data.to_numpy().astype(np.float32)))



