from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch


def plot_model(
    model, ax, lane_names: List[str], range: Tuple[float] = (-6, 10)
) -> None:
    x = np.linspace(*range, 1000).reshape(-1, 1)
    y = model.probability(torch.from_numpy(x).float()).cpu().numpy()
    ax.plot(x, y, color="red")

    y = model.predict_proba(torch.from_numpy(x).float()).cpu().numpy()
    for i, _ in enumerate(lane_names):
        ax.plot(
            x,
            y[:, i],
        )

    for d in model.distributions:
        ax.axvline(d.means[0].cpu(), color="black", linestyle="--")

    ax.set_xlim(*range)
    # make the legend have a white background
    ax.legend(
        ["Observed $d$", *lane_names],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    ax.set_xlabel("$d$ [m]")
    ax.set_ylabel("Probability Density")

