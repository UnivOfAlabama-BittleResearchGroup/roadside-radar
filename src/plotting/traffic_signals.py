from datetime import timedelta
from typing import Dict
import plotly.graph_objects as go
import polars as pl


def open_signal_df(path: str, offset = None) -> pl.DataFrame:
    """Open a traffic signal file as a DataFrame.

    Args:
        path (str): Path to the file.

    Returns:
        pl.DataFrame: The traffic signal file as a DataFrame.
    """
    return (
        pl.scan_csv(path)
        .with_columns(
            pl.col("SignalID").cast(int),
            pl.col("Timestamp").str.strptime(
                dtype=pl.Datetime(time_unit="ns", time_zone="US/Central"),
                format="%Y-%m-%d %H:%M:%S%.f",
            ),
        )
        .with_columns(
            (pl.col('Timestamp') + timedelta(seconds=offset)) if offset else pl.col('Timestamp')
        )
        .rename({"EventParam": "Phase"})
        .sort("Timestamp")
        .with_columns(
            pl.struct(["SignalID", "Phase"]).hash().alias("SignalPhaseID"),
            pl.col("EventCode").replace({1: "G", 8: "Y", 10: "R"}),
        )
        .with_columns(
            pl.col("Timestamp").shift(-1).over("SignalPhaseID").alias("PhaseEnd"),
        )
        # .filter(pl.col("EventCode") == "G")
        .collect(streaming=True)
    )


def add_signals_to_plot(
    df: pl.DataFrame, fig: go.Figure, s_mapping: Dict[int, float]
) -> go.Figure:
    """Add traffic signals to a plot.

    Args:
        df (pl.DataFrame): The traffic signal DataFrame.
        fig (go.Figure): The plot to add the signals to.
        s_mapping (Dict[int, float]): A mapping of signal IDs to the S location of the stop bar.

    Returns:
        go.Figure: The plot with the signals added.
    """

    color_map = {
        "R": "red",
        "Y": "yellow",
        "G": "green",
    }

    for (signal_id, color), _df in df.group_by(["SignalID", "EventCode"]):
        for row in _df.iter_rows(named=True):
            fig.add_trace(
                go.Scatter(
                    x=[row["Timestamp"], row["PhaseEnd"]],
                    y=[s_mapping[signal_id], s_mapping[signal_id]],
                    mode="lines",
                    line=dict(color=color_map[color], width=10),
                    showlegend=False,
                )
            )

    return fig
