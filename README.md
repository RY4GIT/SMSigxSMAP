# smap-drydown
This repository contains code for analyzing soil moisture drydowns from [the Soil Moisture Active Passive (SMAP)](https://smap.jpl.nasa.gov/data/) as detailed in the corresponding manuscript: 

    Araki, R., Morgan, B.E., McMillan, H.K., Caylor, K.K. Nonlinear Soil Moisture Loss Function Reveals Vegetation Responses to Water Availability. Geophysical Research Letters (in review)

## Getting started
1. Clone the repository
```bash
$ git clone git@github.com:RY4GIT/smap-drydown.git
```

2. Create a virtual environment (select appropriate yml file according to your OS). 
```bash
$ cd smap-drydown
$ conda env create -f environment_linux.yml
$ conda activate SMAP
```

3. Download the SMAP and ancillary data from appropriate sources.

4. In the `analysis` directory, create `config.ini`, based on `config_example.ini`

5. Run `analysis\__main__.py`

## Contents

### `data_mng` 
Contains scripts to retrieve & curate input data.

### `analysis`
Contains scripts to implement the drydown analysis and model fits. 
This code has been further refactored in https://github.com/ecohydro/drydowns: checkout if you are interested in. 

### `notebooks`
Contains scripts that are used to test functions or visualise results

## Contact
Ryoko Araki, raraki8150 (at) sdsu.edu