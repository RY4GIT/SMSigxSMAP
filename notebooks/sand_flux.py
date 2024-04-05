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
import seaborn as sns

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
config = Config('config.ini')
cfg = config.config

agent = TowerAgent(cfg, logger=config.logger.getChild(__name__))
# data = Data(cfg, agent.target_EASE_idx[500])
# separator = EventSeparator(agent.cfg, data)


#%%
# proj_dir = '/home/waves/projects/ecohydro-regimes/'
# fd_dir = os.path.join(proj_dir, 'data/FluxData/')
# flx_dir = os.path.join(fd_dir, 'FLUXNET/FULLSET')

# dd_files = os.listdir(os.path.join(flx_dir, 'daily'))

# dd_file = os.path.join(flx_dir, 'daily', 'FLX_AU-Dry_FLUXNET2015_FULLSET_DD_2008-2014_2-4.csv')

# aud = fluxtower.FluxNetTower(dd_file)
# aud.add_vp_cols()
# aud.add_et_cols()

# tower = aud

# Get number of SWC cols per tower
agent.data_ids

meta = fluxtower.flx_tower.META

# Get the number of SWC columns per tower
meta.groupby('SITE_ID')

# (meta[meta.DATAVALUE.str.contains('SWC_')].groupby('SITE_ID').GROUP_ID.count()/2).max()


#%% RUN ALL TOWERS

nprocess = cfg.getint("MULTIPROCESSING", "nprocess")
with mp.Pool(nprocess) as pool:
    output = list(pool.starmap(agent.run, product(agent.data_ids, [True])))
pool.close()
pool.join()


data = [out[0] for out in output if out]
results = pd.concat([out[1] for out in output if out], ignore_index=True)


#%% TESTING ON A SINGLE TOWER

dd_file = os.path.join(cfg.get('PATHS', 'data_dir'), agent._filenames[0])

# tower = fluxtower.FluxNetTower(dd_file)

tower = agent._init_tower('AU-Dry')

cols, col_dict = toweragent.get_cols(tower)
sm_cols = col_dict['SWC']['var_cols']
grps = toweragent.get_grps(tower, sm_cols)

# data_list = [SoilSensorData(agent.cfg, tower, grp) for grp in grps]

# data = data_list[0]

# # for col in sm_cols:
# for data in data_list:
#     # data = SoilSensorData(agent.cfg, tower, col)
#     data.separate_events()

data = []
results = []
for grp in grps:
    out = agent.run_sensor(tower, grp, return_data=True)
    data.append(out[0])
    results.append(out[1])

out = agent.run(tower.id, return_data=True)
# data = out[0][0]
results = out[1]

#%%

turq = '#2195AC'
green = '#81BF24'
maroon = '#80004D'
lav = '#A6A6ED'
red = '#cc3928'


colors = sns.color_palette("husl", 6)


def plot_sm_precip(
    ax, df, swc_cols, p_col, swc_labels, colors=colors, kwargs={'alpha' : 0.8}
):
    ax2 = ax.twinx()
    for i, col in enumerate(swc_cols):
        ax.plot(df.TIMESTAMP, df[col], '.', label=swc_labels[i], color=colors[i], **kwargs)
    ax.set_ylabel('Soil water content (m$^3$ m$^{-3}$)')
    # ax[0].set_yticks(np.arange(0., 1.1, 0.2))
    ax.legend(loc='upper right')

    ax2.bar(df.index, -1*df[p_col], color=turq, alpha=0.6)
    ax2.set_ylabel('Precipitation (mm)')
    ax2.set_yticks(np.arange(0,-190,-30.))
    ax2.set_yticklabels(np.arange(0,190, 30))#[0,30,60,90,120])
    return ax



p_col = 'P_F'
swc_labels = [f"{tower.var_info.get(col).get('HEIGHT')} m" for col in sm_cols]


fig, ax = plt.subplots(1, 1, figsize=(8.0,2.0))
plot_sm_precip(ax, tower.data, sm_cols, p_col, swc_labels)


ed = data.events_df

# Plot vertical line at start of drydown event
for s in ed.start_date:
    ax.axvline(s, color='k', linestyle='--', alpha=0.5)

for e in ed.end_date:
    ax.axvline(e, color='g', linestyle='--', alpha=0.5)


# CASE 1 (Example: 'ZM-Mon'). Lots of bad data. But still can use good data.

# %%
