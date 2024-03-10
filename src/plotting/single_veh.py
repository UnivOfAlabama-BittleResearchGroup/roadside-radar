import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib.pyplot as plt

def plot_vehicle(
    veh_df: pl.DataFrame,
    s_col: str = "s",
    s_velocity_col: str = "s_velocity",
    d_col: str = "d",
    d_velocity_col: str = "d_velocity",
    color: str = "red",
    fig: go.Figure = None,
    data_name: str = None,
    show_d=True,
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

    for i, (name, col, secondary_y) in enumerate(setup):
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


def plot_graph(veh_id: int, sublist: list = None, subgraph: nx.Graph = None) -> None:
    
    # # replace negative weights w/ abs value
    # for u, v, d in subgraph.edges(data=True):
    #     if d["weight"] < 0:
    #         d["weight"] = abs(d["weight"])

            # update the edge weight

    pos = nx.spring_layout(
        subgraph,
    )  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(subgraph, pos, node_size=10)

    # edges
    nx.draw_networkx_edges(subgraph, pos, width=2)
    # nx.draw_networkx_edges(
    #     small_g, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    # )

    # node labels
    nx.draw_networkx_labels(subgraph, pos, font_size=5, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(subgraph, "weight")
    # round the edge weights
    edge_labels = {k: round(v, 2) for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=5)
