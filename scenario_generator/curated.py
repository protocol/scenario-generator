"""
Generates various scenarios that are of interest for analysis.
"""
from typing import Union
import datetime
import numpy as np

import scenario_generator.utils as u

def fit_exponential(x1, y1, x2, y2):
    # Solves y = a*b^x for given points (x1, y1) and (x2, y2)
    # https://math.stackexchange.com/a/3276964
    b = (y2/y1)**(1./(x2-x1))
    a = y1/(b**x1)
    return a, b

def forecast_historical_median_renewal_rate(start_date: datetime.date,
                                            end_date: datetime.date,
                                            forecast_length: int):
    _, historical_renewal_rate = u.get_historical_renewal_rate(start_date, end_date)
    forecasted_renewal_rate = np.ones(forecast_length) * np.median(historical_renewal_rate)
    return forecasted_renewal_rate

def forecast_historical_median_filplus_rate(start_date: datetime.date,
                                            end_date: datetime.date,
                                            forecast_length: int):
    _, historical_filplus_rate = u.get_historical_filplus_rate(start_date, end_date)
    forecasted_filplus_rate = np.ones(forecast_length) * np.median(historical_filplus_rate)
    return forecasted_filplus_rate

def forecast_historical_median_rbonboard_power(start_date: datetime.date,
                                               end_date: datetime.date,
                                               forecast_length: int):
    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, end_date)
    forecasted_rbonboard_pw = np.ones(forecast_length) * np.median(historical_rbonboard_pw)
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

def forecast_optimistic_scenario(forecast_length: int):
    OPTIMISTIC_VAL = 0.99

    start_date = datetime.date(2020, 10, 15) # start of network
    today = datetime.datetime.now().date() - datetime.timedelta(days=1)
    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, today)
    historical_max_rbonboard = np.max(historical_rbonboard_pw)
    rb_onboard_power_forecast = np.ones(forecast_length) * historical_max_rbonboard
    renewal_rate_forecast = np.ones(forecast_length) * OPTIMISTIC_VAL
    fil_plus_rate_forecast = np.ones(forecast_length) * OPTIMISTIC_VAL

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

def forecast_optimistic_scenario_smooth(forecast_length: int):
    OPTIMISTIC_VAL = 0.999

    start_date = datetime.date(2020, 10, 15) # start of network
    today = datetime.datetime.now().date() - datetime.timedelta(days=10)
    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, today)
    historical_max_rbonboard = np.max(historical_rbonboard_pw)
    t = np.arange(forecast_length)
    a, b = fit_exponential(t[0], historical_rbonboard_pw[-1], forecast_length, historical_max_rbonboard)
    rb_onboard_power_forecast = a*np.power(b, t)
    
    _, historical_renewal_rate = u.get_historical_renewal_rate(start_date, today)
    a, b = fit_exponential(t[0], historical_renewal_rate[-1], forecast_length, OPTIMISTIC_VAL)
    renewal_rate_forecast = a*np.power(b, t)

    fil_plus_rate_forecast = forecast_filplus_expcurve_from_date(today, OPTIMISTIC_VAL, forecast_length)

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

def forecast_pessimistic_scenario(forecast_length: int):
    PESSIMISTIC_VAL = 0.01

    rb_onboard_power_forecast = np.ones(forecast_length) * PESSIMISTIC_VAL
    renewal_rate_forecast = np.ones(forecast_length) * PESSIMISTIC_VAL
    fil_plus_rate_forecast = np.ones(forecast_length) * PESSIMISTIC_VAL

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }

def forecast_pessimistic_scenario_smooth(forecast_length: int):
    PESSIMISTIC_VAL = 0.01

    start_date = datetime.date(2020, 10, 15) # start of network
    today = datetime.datetime.now().date() - datetime.timedelta(days=10)
    _, historical_rbonboard_pw = u.get_historical_daily_onboarded_power(start_date, today)
    historical_min_rbonboard = np.min(historical_rbonboard_pw)
    t = np.arange(forecast_length)
    a, b = fit_exponential(t[0], historical_rbonboard_pw[-1], forecast_length, historical_min_rbonboard)
    rb_onboard_power_forecast = a*np.power(b, t)

    _, historical_renewal_rate = u.get_historical_renewal_rate(start_date, today)
    a, b = fit_exponential(t[0], historical_renewal_rate[-1], forecast_length, PESSIMISTIC_VAL)
    renewal_rate_forecast = a*np.power(b, t)

    fil_plus_rate_forecast = forecast_filplus_expcurve_from_date(today, PESSIMISTIC_VAL, forecast_length)

    return {
        'rb_onboard_power': rb_onboard_power_forecast, 
        'renewal_rate': renewal_rate_forecast,
        'filplus_rate': fil_plus_rate_forecast
    }