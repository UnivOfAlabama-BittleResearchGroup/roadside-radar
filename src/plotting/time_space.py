from datetime import timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import polars as pl


def plot_time_space(
    plot_df,
    fig: go.Figure = None,
    s_col: str = "s_filt",
    markers: bool = True,
    color_func: callable = None,
    marker_func: callable = None,
    hoverdata: str = None,
    vehicle_col: str = "vehicle_id",
    every: int = 1,
    **kwargs,
):
    colors = px.colors.qualitative.D3

    if fig is None:
        fig = go.Figure()

    if color_func is None:
        # default color function is just based on the vehicle id
        def color_func(x, i):
            return colors[i % len(colors)]

    if marker_func is None:

        def marker_func(x, i):
            return "circle"

    plot_df_ts = plot_df.sort("epoch_time").to_pandas()

    if "prediction" not in plot_df_ts.columns:
        plot_df_ts["prediction"] = False

    for i, (veh, veh_df) in enumerate(
        plot_df_ts.groupby(
            [
                vehicle_col,
            ]
        )
    ):
        color = color_func(veh_df, i)

        # for obj_id, obj_df in veh_df.groupby(["object_id"]):

        p_df = veh_df.loc[veh_df["prediction"] == True]
        np_df = veh_df.loc[veh_df["prediction"] == False]

        veh_df = veh_df.iloc[::every, :]

        if len(p_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=p_df["epoch_time_cst"],
                    y=p_df[s_col],
                    mode="markers" if markers else "lines",
                    marker=dict(
                        color=color,
                        symbol=marker_func(veh_df, i),
                        opacity=0.3,
                    )
                    if markers
                    else None,
                    line=None if markers else dict(color=color, width=5, dash="dot"),
                    showlegend=False,
                    name=str(veh),
                    hoverinfo="text+name+x+y",
                    hovertext=p_df[hoverdata] if hoverdata is not None else None,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=np_df["epoch_time_cst"],
                y=np_df[s_col],
                mode="markers" if markers else "lines",
                marker=dict(
                    color=color,
                    symbol=marker_func(veh_df, i),
                    opacity=1,
                )
                if markers
                else None,
                line=None
                if markers
                else dict(
                    color=color,
                    width=5,
                ),
                showlegend=False,
                name=str(veh),
                hoverinfo="text+name+x+y",
                hovertext=np_df[hoverdata] if hoverdata is not None else None,
            )
        )

    fig.update_layout(
        height=800,
        width=1200,
        template="ggplot2",
        # update the font size
        font=dict(size=18),
    )

    # reduce the margin on the right side
    fig.update_layout(margin=dict(r=40, t=40, b=40, l=40))

    return fig
