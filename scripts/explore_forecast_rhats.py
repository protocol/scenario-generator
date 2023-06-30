from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import datetime
import matplotlib.dates as mdates

from datetime import timedelta, date, datetime
import time
import os
import pickle

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings('ignore')

from jax import random
import numpyro
import numpyro.distributions as dist

import scenario_generator.mcmc_forecast as mcmc
import scenario_generator.utils as u

import mechafil.minting as minting
import mechafil.data as mecha_data
from mechafil.data import get_historical_network_stats, get_sector_expiration_stats, setup_spacescope
from mechafil.power import forecast_power_stats, build_full_power_stats_df, scalar_or_vector_to_vector
from mechafil.vesting import compute_vesting_trajectory_df
from mechafil.minting import compute_minting_trajectory_df
from mechafil.supply import forecast_circulating_supply_df

from cel_utils import disk_utils

NDEVICES = 4 
numpyro.set_host_device_count(NDEVICES)

# Data starts in filecoin_daily_stats 2021-03-15
# genesis block was in 2020-08-24
# Main net launch was in 2020-10-15

today = datetime.today().date()
DATA_LAG_DAYS = 2
current_date = today - timedelta(days=(DATA_LAG_DAYS))
current_day = (current_date - date(2020, 10, 15)).days

start_date = date(today.year, today.month, 1)  # seed network w/ data uptil the beginning of the month
                                               # to reduce the locking error
start_day = (start_date - date(2020, 10, 15)).days

# Forecast
forecast_lenght = 365 * 5
end_day = current_day + forecast_lenght
end_date = current_date + timedelta(days=forecast_lenght)

print(start_date)
print(current_date)
print(end_date)
duration = 365 # sector duration

# handy constants
EIB = 2 ** 60
PIB = 2 ** 50
TIB = 2 ** 40
GIB = 2 ** 30
SECTOR_SIZE = 32 * GIB
EPOCH_PER_DAY = 2880

auth_config='/Users/kiran/code/filecoin-mecha-twin/kiran_spacescope_auth.json'
setup_spacescope(auth_config)

offline_info_dir = 'explore_rhat_execdate_%s' % (current_date,)
os.makedirs(offline_info_dir, exist_ok=True)

@disk_utils.cache_data(directory=offline_info_dir)
def run_test():
    num_warmup_samps_vec = [1000, 5000]
    num_samples_samps_vec = [1000, 5000]
    seasonality_samps_vec = [2000]
    num_chains_samps_vec = [4]
    train_len_days_vec = [90, 180, 270]
    forecast_list = ['rbp', 'rr', 'fpr']
    train_start_date = current_date
    forecast_length = 365*5
        
    res = mcmc.characterize_mcmc_forecast(
        num_warmup_samps_vec,
        num_samples_samps_vec,
        seasonality_samps_vec,
        num_chains_samps_vec,
        train_len_days_vec,
        forecast_list,
        train_start_date,
        forecast_length
    )
    return res
    
results = run_test()