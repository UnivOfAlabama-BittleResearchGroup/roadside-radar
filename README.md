# Roadside-Radar

This is code to accompany the paper <TODO>


## Installation

- <TODO>


## Dataset Description

A 1 hour snippet of the raw data is avaliable in `data/raw/raw_radar_small.parquet` with the data schema described in `data/fused/data_description.md`

The corresponding fused data is avaliable in `data/fused/fused_3_13_small.parquet` with the data schema described in `data/fused/data_description.md`

## Usage

1. Download data from <TODO>
2. Run `python ./scripts/fuse_radar.py` to associate and fuse the radar data. This script encompases the entire pipeline from data association to fusion, as described in the paper.
   1. The script runs on GPU. There are several "chunk" or batch size parameters hard-coded in the script. You will have to adjust these to fit your GPU memory. PRs to make this more user-friendly are welcome.
   2. Processing the entire March dataset takes lots of RAM. Its possible to process the data in chunks, but this is not implemented in the script, as we had enough memory to process the entire dataset at once. PRs to make this more user-friendly are welcome.
3. Explore the `notebooks` folder for examples of how to use the fused data.
