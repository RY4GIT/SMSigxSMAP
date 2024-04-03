#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2024'

__license__ = 'MIT'
__date__ = 'Sun 31 Mar 24 16:32:29'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           sand_flux.py
Compatibility:  Python 3.7.0
Description:    Description of what program does

URL:            https://

Requires:       list of libraries required

Dev ToDo:       None

AUTHOR:         Bryn Morgan
ORGANIZATION:   University of California, Santa Barbara
Contact:        bryn.morgan@geog.ucsb.edu
Copyright:      (c) Bryn Morgan 2024


"""


#%% IMPORTS



import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
import warnings
import threading
from MyLogger import getLogger
import logging
import multiprocessing as mp
from itertools import product

import fluxtower
from Data import Data
from towerevent import TowerEvent

import configparser
from configparser import ConfigParser
from Agent import Agent
from toweragent import TowerAgent
import toweragent
from EventSeparator import EventSeparator
from towerevent import TowerEvent
from towerseparator import TowerEventSeparator

from towerdata import SoilSensorData
from DrydownModel import DrydownModel

from config import Config

# Create a logger
log = getLogger(__name__)


# cfg = ConfigParser()
# cfg.read("config.ini")

cfg = Config('config.ini').config

agent = TowerAgent(cfg)
# data = Data(cfg, agent.target_EASE_idx[500])
# separator = EventSeparator(agent.cfg, data)


#%%
proj_dir = '/home/waves/projects/ecohydro-regimes/'
fd_dir = os.path.join(proj_dir, 'data/FluxData/')
flx_dir = os.path.join(fd_dir, 'FLUXNET/FULLSET')

dd_files = os.listdir(os.path.join(flx_dir, 'daily'))

dd_file = os.path.join(flx_dir, 'daily', 'FLX_AU-Dry_FLUXNET2015_FULLSET_DD_2008-2014_2-4.csv')

aud = fluxtower.FluxNetTower(dd_file)
aud.add_vp_cols()
aud.add_et_cols()

tower = aud

#%%

nprocess = cfg.getint("MULTIPROCESSING", "nprocess")
with mp.Pool(nprocess) as pool:
    output = list(pool.starmap(agent.run, product(agent.data_ids, [True])))
pool.close()
pool.join()





#%% FUNCTIONS

dd_file = os.path.join(cfg.get('PATHS', 'data_dir'), agent._filenames[-1])

tower = fluxtower.FluxNetTower(dd_file)

cols, col_dict = toweragent.get_cols(tower)
sm_cols = col_dict['SWC']['var_cols']

data_list = [SoilSensorData(agent.cfg, tower, swc_col) for swc_col in sm_cols]



# for col in sm_cols:
for data in data_list:
    # data = SoilSensorData(agent.cfg, tower, col)
    data.separate_events()