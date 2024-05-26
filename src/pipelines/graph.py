import heapq
import multiprocessing as mp
import queue
from typing import List
import polars as pl
import os
import networkx as nx
from itertools import chain, combinations
from scipy.stats import chi2


def worker(veh_queue, permute_df, dims, G, cc, return_queue):
    df = pl.read_parquet(permute_df)  # read temporary data frame
    while True:
        try:
            veh = veh_queue.get_nowait()
        except queue.Empty:
            break
        print(f"Processing vehicle {veh}")
        res = walk_graph_removals(
                G.subgraph(cc[veh]).copy(),
                max_removals=10,
                cutoff=chi2.ppf(0.995, dims),
                df=df,
                return_just_edges=True,
            )
        return_queue.put(
            res
        )

    # return remove_edges

def parallel_process(permute_df, dims, G, cc, ):
    permute_df.write_parquet('tmp.parquet', )  # save dataframe to temporary file
    veh_queue = mp.Queue()
    return_queue = mp.Queue()

    for veh in permute_df["vehicle_index"].unique():
        veh_queue.put(veh)

    jobs = []
    print(mp.cpu_count())
    for i in range(mp.cpu_count()):
        p = mp.Process(target=worker, args=(veh_queue, 'tmp.parquet', dims, G, cc, return_queue))
        jobs.append(p)
        p.start()

    # consume the queue
    
    for proc in jobs:
        proc.join()

    remove_edges = []
    while not return_queue.is_empty():
        remove_edges.append(return_queue.get())
    
    os.remove('tmp.parquet')  # remove temporary file

    return remove_edges



def get_edge_attributes(G, name, cutoff):
    # ...
    edges = G.edges(data=True)
    return (x[-1][name] or (cutoff - 1e-3) for x in edges)


def get_graph_score(
    sub_graph: nx.Graph,
    # df: pl.DataFrame,
    remove_edges: List[int] = (),
    add_edges: List[int] = (),
    reinstate_graph: bool = True,
    big_G: nx.Graph = None,
    cutoff: float = None,
    # combs: List[str] = None,
) -> float:
    for remove in remove_edges:
        sub_graph.remove_edge(*remove)

    for add_edge in add_edges:
        sub_graph.add_edge(*add_edge)

    # return score
    scores = []
    for subgraph in nx.connected_components(sub_graph):
        sg = big_G.subgraph(subgraph)
        try:
            scores.extend(get_edge_attributes(sg, "weight", cutoff))
        except TypeError:
            scores.append(cutoff - 0.0001)

    if reinstate_graph:
        for remove in remove_edges:
            sub_graph.add_edge(*remove)

        for add_edge in add_edges:
            sub_graph.remove_edge(*add_edge)

    return scores


# def check_improvement(old_score, new_score, cutoff):

#     # check that there is improvement over the last score
#     return any(s < cutoff for s in new_score) and np.mean(new_score) < np.mean(old_score)



def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1, 0, -1))


def walk_graph_removals(
    g: nx.Graph,
    cutoff: float = chi2.ppf(0.95, 4),
    max_removals: int = 3,
    big_G: nx.Graph = None,
):
    i = 0
    remove_edges = []
    graph_scores = get_graph_score(
        g,
        big_G=big_G,
        cutoff=cutoff,
    )

    if all(s < cutoff for s in graph_scores):
        return []

    fresh_g = g.copy()

    # sort the edges by their node connectivity
    edges = list(sorted(g.edges, key=lambda x: g.degree(x[0]) + g.degree(x[1])))
    graph_scores = []

    while i == 0 or (
        (i < max_removals) and not all(s < cutoff for s in graph_scores[-1])
    ):
        # for _ in range(max_removals):
        scores = []
        for edge in edges:
            score = get_graph_score(
                # score_func,
                g,
                remove_edges=[edge],
                big_G=big_G,
                cutoff=cutoff,
            )
            # score = np.mean([s for s in score if s is not None])
            # scores.append((score, edge))
            heapq.heappush(scores, (sum(score) / (len(score) or 1), score, edge))

        # # store the scores
        # # keep any that reduce the score underneath the cutoff
        # scores = sorted(scores, key=lambda x: np.mean(x[0]))
        opt_edge = heapq.heappop(scores)

        # check the score
        remove_edges.append(opt_edge[-1])
        graph_scores.append(opt_edge[1])

        # remove what we don't need
        g.remove_edge(*opt_edge[-1])
        edges.remove(opt_edge[-1])

        i += 1

    # now loop and see if we can add any edges back and still satisfy the requirements.
    # This happends for multiple reasons
    # delete_edges = remove_edges.copy()
    pop_edges = []
    # all_valid_edges = set(fresh_g)
    for i, edge in enumerate(remove_edges):
        # edge = list(set(edge).union(all_valid_edges))
        # if not len(edge):
        #     continue
        score = get_graph_score(g, add_edges=[edge], big_G=big_G, cutoff=cutoff)

        if all(s < cutoff for s in score):
            # print("adding_back", edge)
            pop_edges.append(i)
            g.add_edge(*edge)
            # for e in edge:
            # g.add_edge(*e)

    remove_edges = list(nx.difference(fresh_g, g).edges)

    return remove_edges
