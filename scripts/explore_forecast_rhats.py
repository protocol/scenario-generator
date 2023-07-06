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
import argparse
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

def get_plot_data(num_warmup=1000, num_samples=1000, seasonality=2000, num_chains=4, train_days=270):
    key = (num_warmup, num_samples, seasonality, num_chains, train_days)
    res = results[key]
    
    rb_med = np.median(res['rb_onboard_power_pred'], axis=0)
    rr_med = np.median(res['renewal_rate_pred'], axis=0)
    fpr_med = np.median(res['filplus_rate_pred'], axis=0)
    
    rb_rhat = mcmc.check_rhat(res['rb_rhats'])
    ext_rhat = mcmc.check_rhat(res['ext_rhats'])
    expire_rhat = mcmc.check_rhat(res['expire_rhats'])
    deal_onboard_pred_rhats = mcmc.check_rhat(res['deal_onboard_pred_rhats'])
    cc_onboard_pred_rhats = mcmc.check_rhat(res['cc_onboard_pred_rhats'])
    
    rb_rhat_str = '%0.02f' % (rb_rhat,)
    rr_rhat_str = '%0.02f,%0.02f' % (ext_rhat, expire_rhat)
    fpr_rhat_str = '%0.02f,%0.02f' % (deal_onboard_pred_rhats, cc_onboard_pred_rhats)
    
    return rb_med, rb_rhat_str, rr_med, rr_rhat_str, fpr_med, fpr_rhat_str

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Explore forecast rhat values')
    parser.add_argument('--auth', type=str, required=True)
    parser.add_argument('--current-date', type=str, required=False)
    parser.add_argument('--forecast-length', type=int, required=False,
                        help='Number of days to forecast', default=365*5)
    parser.add_argument('--sector-duration', type=int, required=False,
                        help='Avg. Sector Duration', default=365)
    args = parser.parse_args()

    # Set up numpyro
    NDEVICES = 4 
    numpyro.set_host_device_count(NDEVICES)

    # Data starts in filecoin_daily_stats 2021-03-15
    # genesis block was in 2020-08-24
    # Main net launch was in 2020-10-15

    today = datetime.today().date()
    DATA_LAG_DAYS = 2
    if args.current_date is None:    
        current_date = today - timedelta(days=(DATA_LAG_DAYS))
    else:
        current_date = datetime.strptime(args.current_date, '%Y-%m-%d').date()
    current_day = (current_date - date(2020, 10, 15)).days

    start_date = date(today.year, today.month, 1)  # seed network w/ data uptil the beginning of the month
                                                   # to reduce the locking bias
    start_day = (start_date - date(2020, 10, 15)).days

    # Forecast
    forecast_lenght = args.forecast_length
    end_day = current_day + forecast_lenght
    end_date = current_date + timedelta(days=forecast_lenght)

    print(start_date)
    print(current_date)
    print(end_date)
    duration = args.sector_duration

    # handy constants
    EIB = 2 ** 60
    PIB = 2 ** 50
    TIB = 2 ** 40
    GIB = 2 ** 30
    SECTOR_SIZE = 32 * GIB
    EPOCH_PER_DAY = 2880

    if not os.path.isfile(args.auth):
        print('Auth file not found')
        exit(1)
    setup_spacescope(args.auth)

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

    # TODO: update this based on user config for nsamps comparison
    # generate plots of forecasts
    # plot the different predictions for rbp, rr w/ their rhat values
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8))

    nsamp_vec = [1000, 5000]
    for ii, nsamp in enumerate(nsamp_vec):
        for jj, train_hist_days in enumerate([90, 180, 270]):
            rb_med, rb_rhat_str, rr_med, rr_rhat_str, fpr_med, fpr_rhat_str = get_plot_data(nsamp, nsamp, 2000, 4, train_hist_days)
            ax[ii,0].plot(rb_med, label='Hist=%d rhat=%s' % (train_hist_days, rb_rhat_str,))
            ax[ii,1].plot(rr_med, label='Hist=%d rhat=%s' % (train_hist_days, rr_rhat_str,))
            ax[ii,2].plot(fpr_med, label='Hist=%d rhat=%s' % (train_hist_days, rr_rhat_str,))

        ax[ii,0].set_title('RBP[Nsamp=%d]' % (nsamp, ))    
        ax[ii,0].legend()
        ax[ii,1].set_title('RR [%d]' % (nsamp,))
        ax[ii,1].legend()
        ax[ii,2].set_title('FPR [%d]' % (nsamp,))
        ax[ii,2].legend()

    fig.tight_layout()
    plt.show()