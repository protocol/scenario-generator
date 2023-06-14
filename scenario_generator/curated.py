"""
Generates various scenarios that are of interest for analysis.
"""
from typing import Union
import datetime
import numpy as np

import scenario_generator.utils as u

NETWORK_START_DATE = datetime.date(2020, 10, 15)
FAST_NUM_DAYS_TO_TARGET = 15
SLOW_NUM_DAYS_TO_TARGET = 90

def fit_exponential(x1, y1, x2, y2):
    # Solves y = a*b^x for given points (x1, y1) and (x2, y2)
    # https://math.stackexchange.com/a/3276964
    b = (y2/y1)**(1./(x2-x1))
    a = y1/(b**x1)
    return a, b

def smooth_fast_exponential_change_and_plateau(start_value: float, 
                                               target_value: float, 
                                               forecast_length: int, 
                                               num_days_to_target: int):
    t = np.arange(forecast_length)
    a, b = fit_exponential(t[0], start_value, t[num_days_to_target], target_value)
    x1 = a*np.power(b, t[0:num_days_to_target])
    x2 = target_value * np.ones(forecast_length - num_days_to_target)
    x = np.concatenate([x1, x2])
    return x

def linear_increase_and_plateau(start_value: float,
                                target_value: float,
                                forecast_length: int,
                                num_days_to_target: int):
    t = np.arange(forecast_length)
    x1 = np.linspace(start_value, target_value, num_days_to_target)
    x2 = target_value * np.ones(forecast_length - num_days_to_target)
    x = np.concatenate([x1, x2])
    return x

def forecast_historical_median_renewal_rate(start_date: datetime.date,
                                            end_date: datetime.date,
                                            forecast_length: int):
    _, historical_renewal_rate = u.get_historical_renewal_rate(start_date, end_date)
    forecasted_renewal_rate = np.ones(forecast_length) * np.nanmedian(historical_renewal_rate)
    return forecasted_renewal_rate

def forecast_historical_median_filplus_rate(start_date: datetime.date,
                                            end_date: datetime.date,
                                            forecast_length: int):
    _, historical_filplus_rate = u.get_historical_filplus_rate(start_date, end_date)
    forecasted_filplus_rate = np.ones(forecast_length) * np.nanmedian(historical_filplus_rate)
    return forecasted_filplus_rate

def forecast_historical_median_rbonboard_power(start_date: datetime.date,
                                               end_date: datetime.date,
                                               forecast_length: int):
    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, end_date)
    forecasted_rbonboard_pw = np.ones(forecast_length) * np.nanmedian(historical_rbonboard_pw)
    return forecasted_rbonboard_pw

def forecast_rbonboard_power_multiplyfactor(start_value: float,
                                            multiply_factor: float,
                                            forecast_length: int):
    forecasted_rb_onboard_pw = np.ones(forecast_length) * start_value * multiply_factor
    return forecasted_rb_onboard_pw

def forecast_rbonboard_power_exponential_drop(start_value: float,
                                              drop_rate_vec: Union[float, np.array],
                                              drop_time_days_vec: Union[float, np.array],
                                              forecast_length: int):
    pass

def forecast_filplus_expcurve_from_date(start_date: datetime.date,
                                        target_filplus_rate: float,
                                        forecast_length: int):
    historical_start_date = start_date - datetime.timedelta(days=10)
    historical_end_date = start_date
    _, historical_filplus_rate = u.get_historical_filplus_rate(historical_start_date, historical_end_date)
    t = np.arange(forecast_length)
    x1 = t[0]
    y1 = historical_filplus_rate[-1]
    x2 = forecast_length
    y2 = target_filplus_rate
    a, b = fit_exponential(x1, y1, x2, y2)
    forecasted_filplus_rate = a * np.power(b, t)
    return forecasted_filplus_rate

def forecast_optimistic_scenario(forecast_length: int,
                                 pct_max_val: float = 0.9):
    start_date = NETWORK_START_DATE
    today = datetime.datetime.now().date() - datetime.timedelta(days=1)
    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, today)
    historical_max_rbonboard = np.max(historical_rbonboard_pw)
    rb_onboard_power_forecast = np.ones(forecast_length) * historical_max_rbonboard
    renewal_rate_forecast = np.ones(forecast_length) * pct_max_val
    fil_plus_rate_forecast = np.ones(forecast_length) * pct_max_val

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

def forecast_smooth_scenario(forecast_length: int,
                             pct_max_val: float = 0.9,
                             days_to_target: int = 90,
                             rb_onboard_setting: str = 'smooth_pcttarget',
                             renewal_rate_setting: str = 'smooth_pcttarget',
                             filplus_rate_setting: str = 'smooth_pcttarget',
                             start_date: datetime.date = None):
                             
    today = datetime.datetime.now().date() - datetime.timedelta(days=10) if start_date is None else start_date
    
    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(NETWORK_START_DATE, today)
    current_rbonboard_pw = historical_rbonboard_pw[-1]
    if rb_onboard_setting == 'smooth_pcttarget':
        target_onboard = np.max(historical_rbonboard_pw)*pct_max_val
        rb_onboard_power_forecast = smooth_fast_exponential_change_and_plateau(current_rbonboard_pw,
                                                                               target_onboard,
                                                                               forecast_length,
                                                                               days_to_target)    
    elif rb_onboard_setting == 'historical_median':
        target_onboard = np.nanmedian(historical_rbonboard_pw)
        rb_onboard_power_forecast = smooth_fast_exponential_change_and_plateau(current_rbonboard_pw,
                                                                               target_onboard,
                                                                               forecast_length,
                                                                               days_to_target)    

    elif rb_onboard_setting == 'current':
        rb_onboard_power_forecast = np.ones(forecast_length) * current_rbonboard_pw

    _, historical_renewal_rate = u.get_historical_renewal_rate(NETWORK_START_DATE, today)
    current_renewal_rate = historical_renewal_rate[-1]
    if renewal_rate_setting == 'smooth_pcttarget':                                       
        target_renewal_rate = pct_max_val
        renewal_rate_forecast = smooth_fast_exponential_change_and_plateau(current_renewal_rate,
                                                                           target_renewal_rate,
                                                                           forecast_length,
                                                                           days_to_target)
    elif renewal_rate_setting == 'historical_median':
        target_renewal_rate = np.nanmedian(historical_renewal_rate)
        renewal_rate_forecast = smooth_fast_exponential_change_and_plateau(current_renewal_rate,
                                                                           target_renewal_rate,
                                                                           forecast_length,
                                                                           days_to_target)    
    elif renewal_rate_setting == 'current':
        renewal_rate_forecast = np.ones(forecast_length) * current_renewal_rate

    
    # fil_plus_rate_forecast = forecast_filplus_expcurve_from_date(today, pct_max_val, forecast_length)
    _, historical_filplus_rate = u.get_historical_filplus_rate(NETWORK_START_DATE, today)
    current_filplus_rate = historical_filplus_rate[-1]
    if filplus_rate_setting == 'smooth_pcttarget':
        target_filplus_rate = pct_max_val
        fil_plus_rate_forecast = smooth_fast_exponential_change_and_plateau(current_filplus_rate,
                                                                            target_filplus_rate,
                                                                            forecast_length,
                                                                            days_to_target)
    elif filplus_rate_setting == 'historical_median':
        target_filplus_rate = np.nanmedian(historical_filplus_rate)
        fil_plus_rate_forecast = smooth_fast_exponential_change_and_plateau(current_filplus_rate,
                                                                            target_filplus_rate,
                                                                            forecast_length,
                                                                            days_to_target)    

    elif filplus_rate_setting == 'current':
        fil_plus_rate_forecast = np.ones(forecast_length) * current_filplus_rate

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

def forecast_pessimistic_scenario(forecast_length: int,
                                  pct_min_val: float = 0.1):
    rb_onboard_power_forecast = np.ones(forecast_length) * pct_min_val
    renewal_rate_forecast = np.ones(forecast_length) * pct_min_val
    fil_plus_rate_forecast = np.ones(forecast_length) * pct_min_val

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

def forecast_historical_median_scenario(forecast_length: int):
    start_date = NETWORK_START_DATE
    today = datetime.datetime.now().date() - datetime.timedelta(days=10)
    
    renewal_rate_forecast = forecast_historical_median_renewal_rate(start_date,
                                                                    today,
                                                                    forecast_length)
    fil_plus_rate_forecast = forecast_historical_median_filplus_rate(start_date,
                                                                     today,
                                                                     forecast_length)
    rb_onboard_power_forecast = forecast_historical_median_rbonboard_power(start_date,
                                                                           today,
                                                                           forecast_length)

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

def forecast_current_value_scenario(forecast_length: int):
    today = datetime.datetime.now().date() - datetime.timedelta(days=10)
    start_date = today - datetime.timedelta(days=10)
    
    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, today)
    rb_onboard_power_forecast = historical_rbonboard_pw[-1] * np.ones(forecast_length)

    _, historical_renewal_rate = u.get_historical_renewal_rate(start_date, today)
    renewal_rate_forecast = historical_renewal_rate[-1] * np.ones(forecast_length)

    _, historical_filplus_rate = u.get_historical_filplus_rate(start_date, today)
    fil_plus_rate_forecast = historical_filplus_rate[-1] * np.ones(forecast_length)

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

def forecast_onboard_current_rr_high_fpr_median_scenario(forecast_length: int):
    start_date = NETWORK_START_DATE
    today = datetime.datetime.now().date() - datetime.timedelta(days=10)

    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, today)
    rb_onboard_power_forecast = historical_rbonboard_pw[-1] * np.ones(forecast_length)


    _, historical_filplus_rate = u.get_historical_filplus_rate(start_date, today)
    historical_median_filplus_rate = np.nanmedian(historical_filplus_rate)
    current_filplus_rate = historical_filplus_rate[-1]
    fil_plus_rate_forecast = smooth_fast_exponential_change_and_plateau(current_filplus_rate,
                                                                        historical_median_filplus_rate,
                                                                        forecast_length,
                                                                        FAST_NUM_DAYS_TO_TARGET)

    # increase renewal-rate exponentially very quickly to max-value
    MAX_RR = 0.99
    _, historical_renewal_rate = u.get_historical_renewal_rate(start_date, today)
    renewal_rate_forecast = smooth_fast_exponential_change_and_plateau(historical_renewal_rate[-1],
                                                                       MAX_RR,
                                                                       forecast_length,
                                                                       FAST_NUM_DAYS_TO_TARGET)
    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

def forecast_onboard_current_rr_low_fpr_median_scenario(forecast_length: int):
    start_date = NETWORK_START_DATE
    today = datetime.datetime.now().date() - datetime.timedelta(days=10)

    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, today)
    rb_onboard_power_forecast = historical_rbonboard_pw[-1] * np.ones(forecast_length)

    _, historical_filplus_rate = u.get_historical_filplus_rate(start_date, today)
    historical_median_filplus_rate = np.nanmedian(historical_filplus_rate)
    current_filplus_rate = historical_filplus_rate[-1]
    fil_plus_rate_forecast = smooth_fast_exponential_change_and_plateau(current_filplus_rate,
                                                                        historical_median_filplus_rate,
                                                                        forecast_length,
                                                                        FAST_NUM_DAYS_TO_TARGET)

    # increase renewal-rate exponentially very quickly to max-value
    MIN_RR = 0.01
    _, historical_renewal_rate = u.get_historical_renewal_rate(start_date, today)
    renewal_rate_forecast = smooth_fast_exponential_change_and_plateau(historical_renewal_rate[-1],
                                                                       MIN_RR,
                                                                       forecast_length,
                                                                       FAST_NUM_DAYS_TO_TARGET)

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

def forecast_onboard_median_rr_high_fpr_median_scenario(forecast_length: int):
    start_date = NETWORK_START_DATE
    today = datetime.datetime.now().date() - datetime.timedelta(days=10)

    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, today)
    historical_median_rbonboard_pw = np.nanmedian(historical_rbonboard_pw)
    current_rbonboard_pw = historical_rbonboard_pw[-1]
    rb_onboard_power_forecast = smooth_fast_exponential_change_and_plateau(current_rbonboard_pw,
                                                                           historical_median_rbonboard_pw,
                                                                           forecast_length,
                                                                           FAST_NUM_DAYS_TO_TARGET)    

    _, historical_filplus_rate = u.get_historical_filplus_rate(start_date, today)
    historical_median_filplus_rate = np.nanmedian(historical_filplus_rate)
    current_filplus_rate = historical_filplus_rate[-1]
    fil_plus_rate_forecast = smooth_fast_exponential_change_and_plateau(current_filplus_rate,
                                                                        historical_median_filplus_rate,
                                                                        forecast_length,
                                                                        FAST_NUM_DAYS_TO_TARGET)

    # increase renewal-rate exponentially very quickly to max-value
    MAX_RR = 0.99
    _, historical_renewal_rate = u.get_historical_renewal_rate(start_date, today)
    renewal_rate_forecast = smooth_fast_exponential_change_and_plateau(historical_renewal_rate[-1],
                                                                       MAX_RR,
                                                                       forecast_length,
                                                                       FAST_NUM_DAYS_TO_TARGET)
    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

def forecast_onboard_median_rr_low_fpr_median_scenario(forecast_length: int):
    start_date = NETWORK_START_DATE
    today = datetime.datetime.now().date() - datetime.timedelta(days=10)

    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, today)
    historical_median_rbonboard_pw = np.nanmedian(historical_rbonboard_pw)
    current_rbonboard_pw = historical_rbonboard_pw[-1]
    rb_onboard_power_forecast = smooth_fast_exponential_change_and_plateau(current_rbonboard_pw,
                                                                           historical_median_rbonboard_pw,
                                                                           forecast_length,
                                                                           FAST_NUM_DAYS_TO_TARGET)    

    _, historical_filplus_rate = u.get_historical_filplus_rate(start_date, today)
    historical_median_filplus_rate = np.nanmedian(historical_filplus_rate)
    current_filplus_rate = historical_filplus_rate[-1]
    fil_plus_rate_forecast = smooth_fast_exponential_change_and_plateau(current_filplus_rate,
                                                                        historical_median_filplus_rate,
                                                                        forecast_length,
                                                                        FAST_NUM_DAYS_TO_TARGET)

    # increase renewal-rate exponentially very quickly to max-value
    MIN_RR = 0.01
    _, historical_renewal_rate = u.get_historical_renewal_rate(start_date, today)
    renewal_rate_forecast = smooth_fast_exponential_change_and_plateau(historical_renewal_rate[-1],
                                                                       MIN_RR,
                                                                       forecast_length,
                                                                       FAST_NUM_DAYS_TO_TARGET)

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

def forecast_onboard_low_rr_current_fpr_median_scenario(forecast_length: int):
    start_date = NETWORK_START_DATE
    today = datetime.datetime.now().date() - datetime.timedelta(days=10)

    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, today)
    current_rbonboard_pw = historical_rbonboard_pw[-1]
    target_onboard = 1.0
    rb_onboard_power_forecast = smooth_fast_exponential_change_and_plateau(current_rbonboard_pw,
                                                                           target_onboard,
                                                                           forecast_length,
                                                                           FAST_NUM_DAYS_TO_TARGET)    

    _, historical_filplus_rate = u.get_historical_filplus_rate(start_date, today)
    historical_median_filplus_rate = np.nanmedian(historical_filplus_rate)
    current_filplus_rate = historical_filplus_rate[-1]
    fil_plus_rate_forecast = smooth_fast_exponential_change_and_plateau(current_filplus_rate,
                                                                        historical_median_filplus_rate,
                                                                        forecast_length,
                                                                        FAST_NUM_DAYS_TO_TARGET)

    # increase renewal-rate exponentially very quickly to max-value
    _, historical_renewal_rate = u.get_historical_renewal_rate(start_date, today)
    current_renewal_rate = historical_renewal_rate[-1]
    renewal_rate_forecast = current_renewal_rate * np.ones(forecast_length)

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

def forecast_onboard_high_rr_current_fpr_median_scenario(forecast_length: int):
    start_date = NETWORK_START_DATE
    today = datetime.datetime.now().date() - datetime.timedelta(days=10)

    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, today)
    current_rbonboard_pw = historical_rbonboard_pw[-1]
    target_onboard = np.max(historical_rbonboard_pw)
    rb_onboard_power_forecast = smooth_fast_exponential_change_and_plateau(current_rbonboard_pw,
                                                                           target_onboard,
                                                                           forecast_length,
                                                                           FAST_NUM_DAYS_TO_TARGET)    

    _, historical_filplus_rate = u.get_historical_filplus_rate(start_date, today)
    historical_median_filplus_rate = np.nanmedian(historical_filplus_rate)
    current_filplus_rate = historical_filplus_rate[-1]
    fil_plus_rate_forecast = smooth_fast_exponential_change_and_plateau(current_filplus_rate,
                                                                        historical_median_filplus_rate,
                                                                        forecast_length,
                                                                        FAST_NUM_DAYS_TO_TARGET)

    # increase renewal-rate exponentially very quickly to max-value
    _, historical_renewal_rate = u.get_historical_renewal_rate(start_date, today)
    current_renewal_rate = historical_renewal_rate[-1]
    renewal_rate_forecast = current_renewal_rate * np.ones(forecast_length)

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

########
def forecast_onboard_low_rr_median_fpr_median_scenario(forecast_length: int):
    start_date = NETWORK_START_DATE
    today = datetime.datetime.now().date() - datetime.timedelta(days=10)

    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, today)
    current_rbonboard_pw = historical_rbonboard_pw[-1]
    target_onboard = 1.0
    rb_onboard_power_forecast = smooth_fast_exponential_change_and_plateau(current_rbonboard_pw,
                                                                           target_onboard,
                                                                           forecast_length,
                                                                           FAST_NUM_DAYS_TO_TARGET)    

    _, historical_filplus_rate = u.get_historical_filplus_rate(start_date, today)
    historical_median_filplus_rate = np.nanmedian(historical_filplus_rate)
    current_filplus_rate = historical_filplus_rate[-1]
    fil_plus_rate_forecast = smooth_fast_exponential_change_and_plateau(current_filplus_rate,
                                                                        historical_median_filplus_rate,
                                                                        forecast_length,
                                                                        FAST_NUM_DAYS_TO_TARGET)

    # increase renewal-rate exponentially very quickly to median
    _, historical_renewal_rate = u.get_historical_renewal_rate(start_date, today)
    current_renewal_rate = historical_renewal_rate[-1]
    target_renewal_rate = np.nanmedian(historical_renewal_rate)
    renewal_rate_forecast = smooth_fast_exponential_change_and_plateau(current_renewal_rate,
                                                                       target_renewal_rate,
                                                                       forecast_length,
                                                                       FAST_NUM_DAYS_TO_TARGET)

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

def forecast_onboard_high_rr_median_fpr_median_scenario(forecast_length: int):
    start_date = NETWORK_START_DATE
    today = datetime.datetime.now().date() - datetime.timedelta(days=10)

    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, today)
    current_rbonboard_pw = historical_rbonboard_pw[-1]
    target_onboard = np.max(historical_rbonboard_pw)
    rb_onboard_power_forecast = smooth_fast_exponential_change_and_plateau(current_rbonboard_pw,
                                                                           target_onboard,
                                                                           forecast_length,
                                                                           FAST_NUM_DAYS_TO_TARGET)    

    _, historical_filplus_rate = u.get_historical_filplus_rate(start_date, today)
    historical_median_filplus_rate = np.nanmedian(historical_filplus_rate)
    current_filplus_rate = historical_filplus_rate[-1]
    fil_plus_rate_forecast = smooth_fast_exponential_change_and_plateau(current_filplus_rate,
                                                                        historical_median_filplus_rate,
                                                                        forecast_length,
                                                                        FAST_NUM_DAYS_TO_TARGET)

    # increase renewal-rate exponentially very quickly to median
    _, historical_renewal_rate = u.get_historical_renewal_rate(start_date, today)
    current_renewal_rate = historical_renewal_rate[-1]
    target_renewal_rate = np.nanmedian(historical_renewal_rate)
    renewal_rate_forecast = smooth_fast_exponential_change_and_plateau(current_renewal_rate,
                                                                       target_renewal_rate,
                                                                       forecast_length,
                                                                       FAST_NUM_DAYS_TO_TARGET)

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }