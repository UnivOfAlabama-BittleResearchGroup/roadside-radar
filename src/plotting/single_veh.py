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


def plot_graph(
    vehicle_colors,
    bad_edge_color,
    subgraph: nx.Graph = None,
    full_graph: nx.Graph = None,
    ax=None,
    node_num_map=None
) -> None:
    
    vehicle_colors = vehicle_colors.copy()

    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

    def node_num_func(i, x):
        return i if node_num_map is None else  node_num_map[x]


    node_color_mapper = {
        v: (node_num_func(i, v), v, vehicle_colors[j])
        for j, node_list
        in enumerate(nx.connected_components(subgraph))
        for i, v in enumerate(node_list)
    }


    full_sub = nx.subgraph(full_graph, subgraph).copy()

    # remove some of the full_subs if they aren't connected in the subgraph
    for edge in full_sub.edges:
        if not nx.has_path(subgraph, edge[0], edge[1]):
            full_sub.remove_edge(edge[0], edge[1])

    for e in subgraph.edges:
        subgraph[e[0]][e[1]]['weight'] = full_sub[e[0]][e[1]]["weight"]

    pos = nx.spring_layout(
        full_sub,
        k=10
        # weight="weight", iterations=200
    )  # positions for all nodes - seed for reproducibility

    nx.draw_networkx_nodes(
        full_sub, 
        pos,
        nodelist=list(node_color_mapper.keys()),
        node_size=150, 
        ax=ax,
        node_color="#F4FAFC",
        alpha=1,
        linewidths=1.5,
        edgecolors=[x[2] for x in node_color_mapper.values()]
    )

    # edges
    nx.draw_networkx_edges(full_sub, pos, width=2, ax=ax, )

    # node labels
    # if draw_nodes:
    nx.draw_networkx_labels(
        subgraph, 
        pos,
        labels={x[1]: x[0] for x in node_color_mapper.values()},
        font_size=10, 
        font_family="sans-serif", 
        ax=ax,
    )

    # edge weight labels
    edge_labels = nx.get_edge_attributes(subgraph, "weight",)
    # round the edge weights
    edge_labels = {k: round(v, 2) for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=10, ax=ax)

    # if bad_graph is not None:
    bad_edges = nx.difference(full_sub, subgraph)
    for u, v in bad_edges.edges:
        bad_edges[u][v]["weight"] = full_sub[u][v]["weight"]

    nx.draw_networkx_edges(bad_edges, pos, edge_color=bad_edge_color, ax=ax, width=2)
    bad_labels = nx.get_edge_attributes(bad_edges, "weight")
    bad_labels = {k: round(v, 2) for k, v in bad_labels.items()}
    nx.draw_networkx_edge_labels(bad_edges, pos, bad_labels, font_size=10, ax=ax)
    return {v: i for v, (i, *_) in node_color_mapper.items()}
    # plt.title(f"Vehicle {veh_id} Graph")


def letter_subplots(axes=None, letters=None, xoffset=-0.1, yoffset=1.0, **kwargs):
    # directly from https://gist.github.com/bagrow/e3fd0bcfb7e107c0471d657b98ffc19d
    """Add letters to the corners of subplots (panels). By default each axis is
    given an uppercase bold letter label placed in the upper-left corner.
    Args
        axes : list of pyplot ax objects. default plt.gcf().axes.
        letters : list of strings to use as labels, default ["A", "B", "C", ...]
        xoffset, yoffset : positions of each label relative to plot frame
          (default -0.1,1.0 = upper left margin). Can also be a list of
          offsets, in which case it should be the same length as the number of
          axes.
        Other keyword arguments will be passed to annotate() when panel letters
        are added.
    Returns:
        list of strings for each label added to the axes
    Examples:
        Defaults:
            >>> fig, axes = plt.subplots(1,3)
            >>> letter_subplots() # boldfaced A, B, C
        
        Common labeling schemes inferred from the first letter:
            >>> fig, axes = plt.subplots(1,4)        
            >>> letter_subplots(letters='(a)') # panels labeled (a), (b), (c), (d)
        Fully custom lettering:
            >>> fig, axes = plt.subplots(2,1)
            >>> letter_subplots(axes, letters=['(a.1)', '(b.2)'], fontweight='normal')
        Per-axis offsets:
            >>> fig, axes = plt.subplots(1,2)
            >>> letter_subplots(axes, xoffset=[-0.1, -0.15])
            
        Matrix of axes:
            >>> fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
            >>> letter_subplots(fig.axes) # fig.axes is a list when axes is a 2x2 matrix
    """

    # get axes:
    if axes is None:
        axes = plt.gcf().axes
    # handle single axes:
    try:
        iter(axes)
    except TypeError:
        axes = [axes]

    # set up letter defaults (and corresponding fontweight):
    fontweight = "bold"
    ulets = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(axes)])
    llets = list('abcdefghijklmnopqrstuvwxyz'[:len(axes)])
    if letters is None or letters == "A":
        letters = ulets
    elif letters == "(a)":
        letters = [ "({})".format(lett) for lett in llets ]
        fontweight = "normal"
    elif letters == "(A)":
        letters = [ "({})".format(lett) for lett in ulets ]
        fontweight = "normal"
    elif letters in ("lower", "lowercase", "a"):
        letters = llets

    # make sure there are x and y offsets for each ax in axes:
    if isinstance(xoffset, (int, float)):
        xoffset = [xoffset]*len(axes)
    else:
        assert len(xoffset) == len(axes)
    if isinstance(yoffset, (int, float)):
        yoffset = [yoffset]*len(axes)
    else:
        assert len(yoffset) == len(axes)

    # defaults for annotate (kwargs is second so it can overwrite these defaults):
    my_defaults = dict(fontweight=fontweight, fontsize='large', ha="center",
                       va='center', xycoords='axes fraction', annotation_clip=False)
    kwargs = dict( list(my_defaults.items()) + list(kwargs.items()))

    list_txts = []
    for ax,lbl,xoff,yoff in zip(axes,letters,xoffset,yoffset):
        t = ax.annotate(lbl, xy=(xoff,yoff), **kwargs)
        list_txts.append(t)
    return list_txts
