from typing import Union, List
import datetime
from datetime import date, timedelta
import numbers

import numpy as np
from numpy.typing import NDArray
import pandas as pd

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.diagnostics as diag
from operator import attrgetter

import jax.numpy as jnp
import jax.random as random
from numpyro.infer import MCMC, NUTS, Predictive

import scenario_generator.utils as u
from tqdm.auto import tqdm

# https://num.pyro.ai/en/stable/tutorials/time_series_forecasting.html 
def sgt(y, seasonality, future=0):
    """
    TODO!
    """

    # heuristically, standard derivation of Cauchy prior depends on
    # the max value of data
    cauchy_sd = jnp.max(y) / 150
    # NB: priors' parameters are taken from
    # https://github.com/cbergmeir/Rlgt/blob/master/Rlgt/R/rlgtcontrol.R
    nu = numpyro.sample("nu", dist.Uniform(2, 20))
    powx = numpyro.sample("powx", dist.Uniform(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(cauchy_sd))
    offset_sigma = numpyro.sample(
        "offset_sigma", dist.TruncatedCauchy(low=1e-10, loc=1e-10, scale=cauchy_sd)
    )

    coef_trend = numpyro.sample("coef_trend", dist.Cauchy(0, cauchy_sd))
    pow_trend_beta = numpyro.sample("pow_trend_beta", dist.Beta(1, 1))
    # pow_trend takes values from -0.5 to 1
    pow_trend = 1.5 * pow_trend_beta - 0.5
    pow_season = numpyro.sample("pow_season", dist.Beta(1, 1))

    level_sm = numpyro.sample("level_sm", dist.Beta(1, 2))
    s_sm = numpyro.sample("s_sm", dist.Uniform(0, 1))
    init_s = numpyro.sample("init_s", dist.Cauchy(0, y[:seasonality] * 0.3))

    def transition_fn(carry, t):
        level, s, moving_sum = carry
        season = s[0] * level**pow_season
        exp_val = level + coef_trend * level**pow_trend + season
        exp_val = jnp.clip(exp_val, a_min=0)
        # use expected vale when forecasting
        y_t = jnp.where(t >= N, exp_val, y[t])

        moving_sum = (
            moving_sum + y[t] - jnp.where(t >= seasonality, y[t - seasonality], 0.0)
        )
        level_p = jnp.where(t >= seasonality, moving_sum / seasonality, y_t - season)
        level = level_sm * level_p + (1 - level_sm) * level
        level = jnp.clip(level, a_min=0)

        new_s = (s_sm * (y_t - level) / season + (1 - s_sm)) * s[0]
        # repeat s when forecasting
        new_s = jnp.where(t >= N, s[0], new_s)
        s = jnp.concatenate([s[1:], new_s[None]], axis=0)

        omega = sigma * exp_val**powx + offset_sigma
        y_ = numpyro.sample("y", dist.StudentT(nu, exp_val, omega))

        return (level, s, moving_sum), y_

    N = y.shape[0]
    level_init = y[0]
    s_init = jnp.concatenate([init_s[1:], init_s[:1]], axis=0)
    moving_sum = level_init
    with numpyro.handlers.condition(data={"y": y[1:]}):
        _, ys = scan(
            transition_fn, (level_init, s_init, moving_sum), jnp.arange(1, N + future)
        )
    if future > 0:
        numpyro.deterministic("y_forecast", ys[-future:])


def check_rhat(rhat_array: NDArray, threshold: float = 1.05):
    """
    Checks whether the Rhat values are below the threshold.
    """
    rhat_less_th = rhat_array[rhat_array < threshold]
    frac_pass = len(rhat_less_th) / len(rhat_array)
    return frac_pass


def get_rhats(mcmc: MCMC, prob=0.9, exclude_deterministic: bool = True) -> NDArray:
    sites = mcmc._states[mcmc._sample_field]
    if isinstance(sites, dict) and exclude_deterministic:
        state_sample_field = attrgetter(mcmc._sample_field)(mcmc._last_state)
        # XXX: there might be the case that state.z is not a dictionary but
        # its postprocessed value `sites` is a dictionary.
        # TODO: in general, when both `sites` and `state.z` are dictionaries,
        # they can have different key names, not necessary due to deterministic
        # behavior. We might revise this logic if needed in the future.
        if isinstance(state_sample_field, dict):
            sites = {
                k: v
                for k, v in mcmc._states[mcmc._sample_field].items()
                if k in state_sample_field
            }
    summary_dict = diag.summary(sites, prob=prob)
    rhats_vec = []
    for k, v in summary_dict.items():
        if "r_hat" in v:
            rhats = v['r_hat']
            if isinstance(rhats, numbers.Number):
                rhats = [rhats]
            rhats_vec.extend(list(rhats))
    return np.array(rhats_vec)
    
def mcmc_predict(y_train: jnp.array,
                 forecast_length: int,
                 num_warmup_mcmc: int = 500,
                 num_samples_mcmc: int = 100,
                 seasonality_mcmc: int = 1000,
                 num_chains_mcmc: int = 2,
                 verbose: bool = True):
    kernel = NUTS(sgt)
    mcmc = MCMC(kernel, num_warmup=num_warmup_mcmc, num_samples=num_samples_mcmc, num_chains=num_chains_mcmc, progress_bar=verbose)
    mcmc.run(random.PRNGKey(0), y_train, seasonality=seasonality_mcmc)
    if verbose:
        mcmc.print_summary()
    samples = mcmc.get_samples()
    rhat_array = get_rhats(mcmc)

    predictive = Predictive(sgt, samples, return_sites=["y_forecast"])
    predict_dist = predictive(random.PRNGKey(1), y_train, seasonality=seasonality_mcmc, future=forecast_length)[
        "y_forecast"
    ]
    return predict_dist, rhat_array


def forecast_rb_onboard_power(train_start_date: datetime.date, 
                              train_end_date: datetime.date,
                              forecast_length: int,
                              num_warmup_mcmc: int = 500,
                              num_samples_mcmc: int = 100,
                              seasonality_mcmc: int = 1000,
                              num_chains_mcmc: int = 2,
                              verbose: bool = True):
    u.sanity_check_date(train_start_date, err_msg="Specified train_start_date is after today!")
    u.sanity_check_date(train_end_date, err_msg="Specified train_end_date is after today!")

    x, y = u.get_historical_daily_onboarded_power(train_start_date, train_end_date)
    y_train = jnp.array(y)
    u.err_check_train_data(y_train)
    
    # y_scale = y_train.max()
    y_scale = 1
    rb_onboard_power_pred, rhats = mcmc_predict(y_train/y_scale, forecast_length,
                                         num_warmup_mcmc=num_warmup_mcmc, 
                                         num_samples_mcmc=num_samples_mcmc,
                                         seasonality_mcmc=seasonality_mcmc, 
                                         num_chains_mcmc=num_chains_mcmc,
                                         verbose=verbose)
    rb_onboard_power_pred *= y_scale
    rb_onboard_power_pred = jnp.clip(rb_onboard_power_pred, 0)
    
    forecast_start_date = train_end_date + datetime.timedelta(days=1)
    forecast_date_vec = u.make_forecast_date_vec(forecast_start_date, forecast_length)
    return forecast_date_vec, rb_onboard_power_pred, x, y, rhats

def forecast_extensions(train_start_date: datetime.date, 
                        train_end_date: datetime.date,
                        forecast_length: int,
                        num_warmup_mcmc: int = 500,
                        num_samples_mcmc: int = 100,
                        seasonality_mcmc: int = 1000,
                        num_chains_mcmc: int = 2,
                        historical_extensions_fp: str = None,
                        verbose: bool = True):
    u.sanity_check_date(train_start_date, err_msg="Specified train_start_date is after today!")
    u.sanity_check_date(train_end_date, err_msg="Specified train_end_date is after today!")

    if historical_extensions_fp is None:
        x, y = u.get_historical_extensions(train_start_date, train_end_date)
    else:
        x, y = u.get_historical_expirations_offline(train_start_date, train_end_date, historical_extensions_fp)
    y_train = jnp.array(y)
    u.err_check_train_data(y_train)

    extensions_pred, rhats = mcmc_predict(y_train, forecast_length,
                                   num_warmup_mcmc=num_warmup_mcmc, 
                                   num_samples_mcmc=num_samples_mcmc,
                                   seasonality_mcmc=seasonality_mcmc, 
                                   num_chains_mcmc=num_chains_mcmc,
                                   verbose=verbose)
    extensions_pred = jnp.clip(extensions_pred, 0)
    
    forecast_start_date = train_end_date + datetime.timedelta(days=1)
    forecast_date_vec = u.make_forecast_date_vec(forecast_start_date, forecast_length)
    return forecast_date_vec, extensions_pred, x, y, rhats

def forecast_expirations(train_start_date: datetime.date, 
                         train_end_date: datetime.date,
                         forecast_length: int,
                         num_warmup_mcmc: int = 500,
                         num_samples_mcmc: int = 100,
                         seasonality_mcmc: int = 1000,
                         num_chains_mcmc: int = 2,
                         historical_expirations_fp: str = None,
                         verbose: bool = True):
    u.sanity_check_date(train_start_date, err_msg="Specified train_start_date is after today!")
    u.sanity_check_date(train_end_date, err_msg="Specified train_end_date is after today!")

    if historical_expirations_fp is None:
        x, y = u.get_historical_expirations(train_start_date, train_end_date)
    else:
        x, y = u.get_historical_expirations_offline(train_start_date, train_end_date, historical_expirations_fp)
    y_train = jnp.array(y)
    u.err_check_train_data(y_train)

    expire_pred, rhats = mcmc_predict(y_train, forecast_length,
                               num_warmup_mcmc=num_warmup_mcmc, 
                               num_samples_mcmc=num_samples_mcmc,
                               seasonality_mcmc=seasonality_mcmc, 
                               num_chains_mcmc=num_chains_mcmc,
                               verbose=verbose)
    expire_pred = jnp.clip(expire_pred, 0)
    
    forecast_start_date = train_end_date + datetime.timedelta(days=1)
    forecast_date_vec = u.make_forecast_date_vec(forecast_start_date, forecast_length)
    return forecast_date_vec, expire_pred, x, y, rhats

def forecast_renewal_rate(train_start_date: datetime.date, 
                          train_end_date: datetime.date,
                          forecast_length: int,
                          num_warmup_mcmc: int = 500,
                          num_samples_mcmc: int = 100,
                          seasonality_mcmc: int = 1000,
                          num_chains_mcmc: int = 2,
                          historical_extensions_fp: str = None,
                          historical_expirations_fp: str = None,
                          verbose: bool = True):
    u.sanity_check_date(train_start_date, err_msg="Specified train_start_date is after today!")
    u.sanity_check_date(train_end_date, err_msg="Specified train_end_date is after today!")

    forecast_date_vec, extensions_pred, x_extend, y_extend, ext_rhats = forecast_extensions(train_start_date, 
                                                                                 train_end_date,
                                                                                 forecast_length,
                                                                                 num_warmup_mcmc = num_warmup_mcmc,
                                                                                 num_samples_mcmc = num_samples_mcmc,
                                                                                 seasonality_mcmc = seasonality_mcmc,
                                                                                 num_chains_mcmc = num_chains_mcmc,
                                                                                 historical_extensions_fp = historical_extensions_fp,
                                                                                 verbose = verbose)
    _, expire_pred, x_expire, y_expire, expire_rhats = forecast_expirations(train_start_date, 
                                                              train_end_date,
                                                              forecast_length,
                                                              num_warmup_mcmc = num_warmup_mcmc,
                                                              num_samples_mcmc = num_samples_mcmc,
                                                              seasonality_mcmc = seasonality_mcmc,
                                                              num_chains_mcmc = num_chains_mcmc,
                                                              historical_expirations_fp = historical_expirations_fp,
                                                              verbose = verbose)
    if not x_extend.equals(x_expire):
        raise ValueError("Unable to get the same amount of data for extensions and expirations!")
    renewal_rate_historical = y_extend / (y_extend + y_expire)

    renewal_rate_pred = extensions_pred / (extensions_pred + expire_pred)
    renewal_rate_pred = jnp.clip(jnp.nan_to_num(renewal_rate_pred, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    return forecast_date_vec, renewal_rate_pred, x_extend, renewal_rate_historical, ext_rhats, expire_rhats


def forecast_filplus_rate(train_start_date: datetime.date, 
                          train_end_date: datetime.date,
                          forecast_length: int,
                          num_warmup_mcmc: int = 500,
                          num_samples_mcmc: int = 100,
                          seasonality_mcmc: int = 1000,
                          num_chains_mcmc: int = 2,
                          historical_deals_onboard_fp: str = None,
                          verbose: bool = True):
    """
    1. forecast deal_onboard --> deal_onboard_dist
    2. find cc_onboard = rawbyte_onboard - deal_onboard
    3. forecast cc_onboard --> cc_onboard_dist
    4. find fil_plus_rate_dist =  deal_onboard_dist / (cc_onboard_dist + deal_onboard_dist)
    """
    u.sanity_check_date(train_start_date, err_msg="Specified train_start_date is after today!")
    u.sanity_check_date(train_end_date, err_msg="Specified train_end_date is after today!")

    if historical_deals_onboard_fp is None:
        x_deal_onboard_train, y = u.get_historical_deals_onboard(train_start_date, train_end_date)
    else:
        x_deal_onboard_train, y = u.get_historical_deals_onboard_offline(train_start_date, train_end_date, historical_deals_onboard_fp)
    y_deal_onboard_train = jnp.array(y)
    u.err_check_train_data(y_deal_onboard_train)

    x_rb_onboard_train, y_rb_onboard_train = \
        u.get_historical_daily_onboarded_power(train_start_date, train_end_date)
    train_start_date = pd.to_datetime(max(x_deal_onboard_train.values[0], x_rb_onboard_train.values[0]))
    train_end_date = pd.to_datetime(min(x_deal_onboard_train.values[-1], x_rb_onboard_train.values[-1]))

    ii_start = np.where(train_start_date==x_rb_onboard_train.values)[0][0]
    ii_end = np.where(train_end_date==x_rb_onboard_train.values)[0][0]
    x_rb_onboard_train = x_rb_onboard_train[ii_start:ii_end]
    y_rb_onboard_train = y_rb_onboard_train[ii_start:ii_end]

    ii_start = np.where(train_start_date==x_deal_onboard_train.values)[0][0]
    ii_end = np.where(train_end_date==x_deal_onboard_train.values)[0][0]
    x_deal_onboard_train = x_deal_onboard_train[ii_start:ii_end]
    y_deal_onboard_train = y_deal_onboard_train[ii_start:ii_end]

    # y_deal_onboard_scale = y_deal_onboard_train.max()
    y_deal_onboard_scale = 1
    deal_onboard_pred, deal_onboard_pred_rhats = mcmc_predict(y_deal_onboard_train/y_deal_onboard_scale, forecast_length,
                                     num_warmup_mcmc=num_warmup_mcmc, 
                                     num_samples_mcmc=num_samples_mcmc,
                                     seasonality_mcmc=seasonality_mcmc, 
                                     num_chains_mcmc=num_chains_mcmc,
                                     verbose=verbose)
    forecast_start_date = train_end_date + datetime.timedelta(days=1)
    forecast_date_vec = u.make_forecast_date_vec(forecast_start_date, forecast_length)
    deal_onboard_pred *= y_deal_onboard_scale

    y_cc_onboard_train = jnp.array(y_rb_onboard_train - y_deal_onboard_train)
    # y_cc_onboard_scale = y_cc_onboard_train.max()
    y_cc_onboard_scale = 1
    cc_onboard_pred, cc_onboard_pred_rhats = mcmc_predict(y_cc_onboard_train/y_cc_onboard_scale, forecast_length,
                                   num_warmup_mcmc=num_warmup_mcmc, 
                                   num_samples_mcmc=num_samples_mcmc,
                                   seasonality_mcmc=seasonality_mcmc, 
                                   num_chains_mcmc=num_chains_mcmc,
                                   verbose=verbose)
    cc_onboard_pred *= y_cc_onboard_scale

    xx = x_rb_onboard_train
    yy = y_deal_onboard_train / (y_cc_onboard_train + y_deal_onboard_train)
    filplus_rate_pred = deal_onboard_pred / (cc_onboard_pred + deal_onboard_pred)
    filplus_rate_pred = jnp.clip(filplus_rate_pred, 0.0, 1.0)
    return forecast_date_vec, filplus_rate_pred, xx, yy, deal_onboard_pred_rhats, cc_onboard_pred_rhats

def characterize_mcmc_forecast(
        num_warmup_samps_vec: Union[List[int], NDArray] = None,
        num_samples_samps_vec: Union[List[int], NDArray] = None,
        seasonality_samps_vec: Union[List[int], NDArray] = None,
        num_chains_samps_vec: Union[List[int], NDArray] = None,
        train_len_days_vec: Union[List[int], NDArray] = None,
        forecast_list: List[str] = None,
        train_end_date: date = None,
        forecast_length: int = 365*5,
):
    def run_forecast(train_start_date, train_end_date, forecast_length,
                     num_warmup_mcmc, num_samples_mcmc, seasonality_mcmc, num_chains_mcmc):
        verbose = False
        forecast_results = {}
        if 'rbp' in forecast_list:
            forecast_rb_date_vec, rb_onboard_power_pred, historical_rb_date, historical_rb, rb_rhats = \
            forecast_rb_onboard_power(
                train_start_date, 
                train_end_date,
                forecast_length,
                num_warmup_mcmc = num_warmup_mcmc,
                num_samples_mcmc = num_samples_mcmc,
                seasonality_mcmc = seasonality_mcmc,
                num_chains_mcmc = num_chains_mcmc,
                verbose=verbose
            )
            rbp_results = {
                'forecast_rb_date_vec': forecast_rb_date_vec, 
                'rb_onboard_power_pred': rb_onboard_power_pred, 
                'historical_rb_date': historical_rb_date, 
                'historical_rb': historical_rb, 
                'rb_rhats': rb_rhats
            }
            forecast_results.update(rbp_results)

        if 'rr' in forecast_list:
            forecast_rr_date_vec, renewal_rate_pred, historical_rr_date , historical_rr, ext_rhats, expire_rhats = \
            forecast_renewal_rate(
                train_start_date, 
                train_end_date,
                forecast_length,
                num_warmup_mcmc = num_warmup_mcmc,
                num_samples_mcmc = num_samples_mcmc,
                seasonality_mcmc = seasonality_mcmc,
                num_chains_mcmc = num_chains_mcmc,
                verbose=verbose
            )
            rr_results = {
                'forecast_rr_date_vec': forecast_rr_date_vec, 
                'renewal_rate_pred': renewal_rate_pred, 
                'historical_rr_date': historical_rr_date, 
                'historical_rr': historical_rr, 
                'ext_rhats': ext_rhats, 
                'expire_rhats': expire_rhats
            }
            forecast_results.update(rr_results)

        if 'fpr' in forecast_list:
            forecast_fpr_date_vec, filplus_rate_pred, historical_fpr_date, historical_fpr, deal_onboard_pred_rhats, cc_onboard_pred_rhats = \
            forecast_filplus_rate(
                train_start_date, 
                train_end_date,
                forecast_length,
                num_warmup_mcmc = num_warmup_mcmc,
                num_samples_mcmc = num_samples_mcmc,
                seasonality_mcmc = seasonality_mcmc,
                num_chains_mcmc = num_chains_mcmc,
                verbose=verbose
            )
            fpr_results = {
                'forecast_fpr_date_vec': forecast_fpr_date_vec, 
                'filplus_rate_pred': filplus_rate_pred, 
                'historical_fpr_date': historical_fpr_date, 
                'historical_fpr': historical_fpr, 
                'deal_onboard_pred_rhats': deal_onboard_pred_rhats, 
                'cc_onboard_pred_rhats': cc_onboard_pred_rhats
            }
            forecast_results.update(fpr_results)
        return forecast_results
    
    if num_warmup_samps_vec is None:
        num_warmup_samps_vec = [1000, 2500, 5000]
    if num_samples_samps_vec is None:
        num_samples_samps_vec = [1000, 2500, 5000]
    if seasonality_samps_vec is None:
        seasonality_samps_vec = [2000]
    if num_chains_samps_vec is None:
        num_chains_samps_vec = [2, 4]
    if train_len_days_vec is None:
        train_len_days_vec = [90, 180, 270]
    if forecast_list is None:
        forecast_list = ['rbp', 'rr', 'fpr']

    if train_end_date is None:
        train_end_date = date.today() - timedelta(days=3)
    
    n = len(num_warmup_samps_vec)*len(num_samples_samps_vec)*len(seasonality_samps_vec)*len(num_chains_samps_vec)*len(train_len_days_vec)
    pbar = tqdm(total=n)
    characterization_results = {}
    for num_warmup_mcmc in num_warmup_samps_vec:
        for num_samples_mcmc in num_samples_samps_vec:
            for seasonality_mcmc in seasonality_samps_vec:
                for num_chains_mcmc in num_chains_samps_vec:
                    for train_len_days in train_len_days_vec:
                        train_start_date = train_end_date - timedelta(days=train_len_days)
                        forecast_results = run_forecast(train_start_date, train_end_date, forecast_length, num_warmup_mcmc, num_samples_mcmc, seasonality_mcmc, num_chains_mcmc)
                        characterization_results[(num_warmup_mcmc, num_samples_mcmc, seasonality_mcmc, num_chains_mcmc, train_len_days)] = forecast_results
                        
                        pbar.update(1)
                        
    return characterization_results