import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_vehicle(
    veh_df: pl.DataFrame,
    s_col: str = "s",
    s_velocity_col: str = "s_velocity",
    d_col: str = "d",
    d_velocity_col: str = "d_velocity",
    color: str = "red",
    fig: go.Figure = None,
    data_name: str = None,
    show_d = True,
    row_mod: int = 0,
    marker_size: int = 5,
    
) -> go.Figure:
    
    if show_d:
        specs = [
            [{"secondary_y": True}],
            [{"secondary_y": True}],
        ]
    else:
        specs = [
            [{"secondary_y": True}],
        ] 


    if fig is None:
        fig = make_subplots(
            rows=2 if show_d else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            # subplot_titles=(
            #     "Vehicle D",
            #     "Vehicle S",
            # ),
            # add a secondary y axis to the velocity plots
            specs=specs,
        )

    setup = [
            ("s", s_col, False),
            ("s_velocity", s_velocity_col, True),
        ] + ([("d", d_col, False), ("d_velocity", d_velocity_col, True)] if show_d else [])

    for i, (name, col, secondary_y) in enumerate(
        setup
    ):
        fig.add_trace(
            go.Scatter(
                x=veh_df["epoch_time_cst"].cast(str),
                y=veh_df[col],
                mode="markers",
                name=data_name if data_name is not None else name,
                marker=dict(color=color, size=marker_size),
                showlegend=(i == 0 and data_name is not None),
                # line=dict(color=color, dash="solid" if secondary_y else "dot"),
            ),
            secondary_y=secondary_y,
            row=i // 2 + 1 + row_mod,
            col=1,
        )

    return fig