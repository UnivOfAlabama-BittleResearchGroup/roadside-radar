# Roadside-Radar

This is code to accompany the paper <TODO>

## Internal

- [ ] Add a link to the paper
- [ ] Raw data currently stored in `DOECV2X/Working_Papers/radar-trajectory-extraction/raw_data/`
- [ ] Cached data currently stored in `/media/HDD/max/radar-data/<month>/cache`
- [ ] Create a dataset description
- [ ] Create a demo dataset
- [ ] Add speed back to the small dataset

## Installation

- [ ]

## Usage

1. Download data from <TODO>
2. Run `python ./scripts/fuse_radar.py` to associate and fuse the radar data. This script encompases the entire pipeline from data association to fusion, as described in the paper.
   1. The script runs on GPU. There are several "chunk" or batch size parameters hard-coded in the script. You will have to adjust these to fit your GPU memory. PRs to make this more user-friendly are welcome.
   2. Processing the entire March dataset takes >200 GB of RAM. Its possible to process the data in chunks, but this is not implemented in the script, as we had enough memory to process the entire dataset at once. PRs to make this more user-friendly are welcome.
3. 