# smap-drydown
This repository contains code for analyzing soil moisture drydowns from [the Soil Moisture Active Passive (SMAP)](https://smap.jpl.nasa.gov/data/) as detailed in the corresponding manuscript: 

    Araki, R., Morgan, B.E., McMillan, H.K., Caylor, K.K. Nonlinear Soil Moisture Loss Function Reveals Vegetation Responses to Water Availability. Submitted to Geophysical Research Letters (in review)

## Getting started
1. Clone the repository
```bash
$ git clone git@github.com:RY4GIT/smap-drydown.git
```

2. Create a virtual environment (select an appropriate yml file according to your OS). 
```bash
$ cd smap-drydown
$ conda env create -f environment_linux.yml
$ conda activate SMAP
```

3. Download the SMAP and ancillary data from appropriate sources using scripts in `data_mng`

4. In the `analysis` directory, create `config.ini`, based on `config_example.ini`

5. Run `analysis\__main__.py`

6. Visualize the results using scripts in `notebooks`. The results file is large (~130 MB) and is therefore available upon request.

## Contents

### `analysis`
Contains scripts to implement the drydown analysis and model fits. 

The functions for loss calculations and drydown models are contained in `DrydownModel.py`. 
This code has been further refactored in https://github.com/ecohydro/drydowns; check it out if you are interested.


### `data_mng` 
Contains scripts to retrieve and curate input data. 

All data are pre-curated in "datarods" format, which stores the data as a long time series at a single SMAP grid.

- SMAP soil moisture data
    - Download data using `retrieve_NSIDC_Data_SPL3SMP.ipynb`
    - Preprocess data using `create_datarods_SPL3SMP.ipynb`
- SMAP precipitation data
    - Download data using `retrieve_NSIDC_Data_SPL4SMGP.ipynb`
    - Preprocess data using `create_datarods_SPL4SMGP.py`
- dPET (Singer et al., 2020) data
    - Download daily data from [the website](http://doi.org/10.5523/bris.qb8ujazzda0s2aykkv0oq0ctp)
    - Preprocess data using `create_datarods_PET.py`
- SMAP ancillary data
    - Download data from [the website](https://doi.org/10.5067/HB8BPJ13TDQJ)
    - Preprocess data using `read_ancillary_landcover_data.ipynb`
    - After obtaining precipitation and PET data, run `calc_aridityindex.py`
- Rangeland data
    - Download data using `retrieve_rangeland_data.sh`
    - Preprocess data using `read_rangeland_data.py`
- Other utilities
    - `retrieve_NSIDC_Data_datacheck.ipynb`: check if all the data are downloaded from NSIDC
    - `create_datarods_datacheck.ipynb`: check if all the data are preprocessed
    - `identify_unusable_grid.ipynb`: identify grids located on open water


### `notebooks`
Contains scripts used to test functions or visualize the models and results
- `figs_stats_datapreprocess.py`: Preprocess result files to reduce execution time
- `figs_method.py` & `figs_method_tau.py`: Visualize the loss functions and drydown models
- `figs_stats.py` and `figs_stats_rev.py`: Visualize the results
- `figs_drydown.py`: Plot observed and modeled drydown curves


## Contact
Ryoko Araki, raraki8159 (at) sdsu.edu