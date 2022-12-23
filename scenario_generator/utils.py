import datetime
import requests

import pandas as pd
import numpy as np

import jax.numpy as jnp

from mechafil.data import query_starboard_daily_power_onboarded, \
                          query_starboard_sector_expirations

PIB = 2**50

def sanity_check_date(date_in: datetime.date, err_msg = None):
    today = datetime.datetime.now().date()
    if date_in > today:
        err_msg_out = err_msg if err_msg is not None else "Supplied date is after today: %s" % (today, )
        raise ValueError(err_msg)


def err_check_train_data(y_train: jnp.array):
    # TODO: improve this to check for "anomalous" data such as a consecutive series of 0's
    #  rather than this basic version
    if len(jnp.where(y_train == 0)[0]) > 3:
        raise ValueError("Starboard data may not be fully populated, \
                          because some onboarded-rb-power data is downloaded as 0!")

def make_forecast_date_vec(forecast_start_date: datetime.date, forecast_length: int):
    forecast_date_vec = [datetime.timedelta(days=int(x))+forecast_start_date for x in np.arange(forecast_length)]
    return forecast_date_vec

def get_historical_daily_onboarded_power(start_date: datetime.date,
                                         end_date: datetime.date):
    sanity_check_date(start_date, err_msg="Specified start_date is after today!")
    sanity_check_date(end_date, err_msg="Specified end_date is after today!")

    onboards_df = query_starboard_daily_power_onboarded(start_date, end_date)
    t_vec = pd.to_datetime(onboards_df.date)
    rb_onboard_vec = onboards_df['day_onboarded_rb_power_pib'].values
    return t_vec, rb_onboard_vec

def get_historical_renewal_rate(start_date: datetime.date,
                                end_date: datetime.date):
    sector_expirations_df = query_starboard_sector_expirations(start_date, end_date)
    t_vec = pd.to_datetime(sector_expirations_df.date)

    historical_renewal_rate = sector_expirations_df['extended_rb'] / (sector_expirations_df['extended_rb'] + sector_expirations_df['expired_rb'] + sector_expirations_df['open_rb'])
    historical_renewal_rate = historical_renewal_rate.values
    
    return t_vec, historical_renewal_rate

def get_historical_extensions(start_date: datetime.date,
                              end_date: datetime.date):
    df = pd.read_csv('offline_info/Scheduled_Expiration_by_Date_Breakdown_in_PiB.csv')
    df = df[df.stateTime <= str(end_date)]
    # NOTE: this can be removed when we upgrade this to get data directly from starboard
    num_days_train = end_date - start_date
    num_days_train = int(num_days_train.days)
    df = df.iloc[-num_days_train:]

    t_vec = pd.to_datetime(df.stateTime)
    extend_vec = df['Extend'].values

    return t_vec, extend_vec

def get_historical_extensions_online(start_date: datetime.date,
                              end_date: datetime.date):
    url = f"https://observable-api.starboard.ventures/getdata/sectors_schedule_expiration_full?start={str(start_date)}&end={str(end_date)}"
    r = requests.get(url)
    # Put data in dataframe
    extensions_df_starboard = pd.DataFrame(r.json()["data"])
    extensions_df_starboard['date'] = pd.to_datetime(extensions_df_starboard['stat_date'])
    for c in extensions_df_starboard.columns:
        if 'date' not in c:
            extensions_df_starboard[c] = extensions_df_starboard[c].astype(float)
    

    t_vec = extensions_df_starboard['date']
    extend_vec = extensions_df_starboard['extended_bytes'].values/PIB

    return t_vec, extend_vec


def get_historical_expirations(start_date: datetime.date,
                               end_date: datetime.date):
    df = pd.read_csv('offline_info/Scheduled_Expiration_by_Date_Breakdown_in_PiB.csv')
    df = df[df.stateTime <= str(end_date)]
    # NOTE: this can be removed when we upgrade this to get data directly from starboard
    num_days_train = end_date - start_date
    num_days_train = int(num_days_train.days)
    df = df.iloc[-num_days_train:]

    t_vec = pd.to_datetime(df.stateTime)
    expire_vec = df['Expired'].values

    return t_vec, expire_vec

def get_historical_expirations_online(start_date: datetime.date,
                               end_date: datetime.date):
    url = f"https://observable-api.starboard.ventures/getdata/sectors_schedule_expiration_full?start={str(start_date)}&end={str(end_date)}"
    r = requests.get(url)
    # Put data in dataframe
    expirations_df_starboard = pd.DataFrame(r.json()["data"])
    expirations_df_starboard['date'] = pd.to_datetime(expirations_df_starboard['stat_date'])
    for c in expirations_df_starboard.columns:
        if 'date' not in c:
            expirations_df_starboard[c] = expirations_df_starboard[c].astype(float)

    t_vec = expirations_df_starboard['date']
    expire_vec = expirations_df_starboard['expired_bytes'].values/PIB
    return t_vec, expire_vec

def get_historical_deals_onboard(start_date: datetime.date,
                                 end_date: datetime.date):
    df = pd.read_csv('offline_info/Daily_Active_Deal_TiB_Change_Breakdown.csv')
    df['deals_onboard'] = df['New Active Deal'] / 1024
    df = df[df.stateTime <= str(end_date)]
    # NOTE: this can be removed when we upgrade this to get data directly from starboard
    num_days_train = end_date - start_date
    num_days_train = int(num_days_train.days)
    df = df.iloc[-num_days_train:]

    t_vec = pd.to_datetime(df.stateTime)
    deals_onboard_vec = df.deals_onboard.values

    return t_vec, deals_onboard_vec

def get_historical_deals_onboard_online(start_date: datetime.date,
                                 end_date: datetime.date):
    url = f"https://observable-api.starboard.ventures/getdata/deal-states-aggregate-daily?start={str(start_date)}&end={str(end_date)}"
    r = requests.get(url)
    # Put data in dataframe
    deals_onboard_df_starboard = pd.DataFrame(r.json()["data"])
    deals_onboard_df_starboard['date'] = pd.to_datetime(deals_onboard_df_starboard['stat_date'])

    for c in deals_onboard_df_starboard.columns:
        if 'date' not in c:
            deals_onboard_df_starboard[c] = deals_onboard_df_starboard[c].astype(float)


    t_vec = deals_onboard_df_starboard['date']
    deals_onboard_vec = deals_onboard_df_starboard['activated_deals_regular_bytes'].values + \
                        deals_onboard_df_starboard['activated_deals_verified_bytes'].values

    return t_vec, deals_onboard_vec


def get_historical_filplus_rate(start_date: datetime.date,
                                end_date: datetime.date):
    rb_onboard_t_vec, rb_onboard_vec = get_historical_daily_onboarded_power(start_date, end_date)
    deal_onboard_t_vec, deal_onboard_vec = get_historical_deals_onboard(start_date, end_date)

    # align the data
    start_date_aligned = pd.to_datetime(max(deal_onboard_t_vec.values[0], rb_onboard_t_vec.values[0]))
    end_date_aligned = pd.to_datetime(min(deal_onboard_t_vec.values[-1], rb_onboard_t_vec.values[-1]))

    ii_start = np.where(start_date_aligned==rb_onboard_t_vec.values)[0][0]
    ii_end = np.where(end_date_aligned==rb_onboard_t_vec.values)[0][0]
    rb_onboard_vec_aligned = rb_onboard_vec[ii_start:ii_end]

    ii_start = np.where(start_date_aligned==deal_onboard_t_vec.values)[0][0]
    ii_end = np.where(end_date_aligned==deal_onboard_t_vec.values)[0][0]
    deal_onboard_vec_aligned = deal_onboard_vec[ii_start:ii_end]

    t_vec_aligned = deal_onboard_t_vec[ii_start:ii_end]
    historical_filplus_rate = deal_onboard_vec_aligned/rb_onboard_vec_aligned
    return t_vec_aligned, historical_filplus_rate