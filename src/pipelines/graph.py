import multiprocessing as mp
import queue
import polars as pl
from tqdm import tqdm
import os
from scipy.stats import chi2

from src.pipelines.association import walk_graph_removals

def worker(veh_queue, permute_df, dims, G, cc, return_queue):
    df = pl.read_parquet(permute_df)  # read temporary data frame
    remove_edges = []
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