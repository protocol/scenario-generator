import numpy as np


def gbm_forecast(x, forecast_length, dt=1, seed=1234):
    """
    Creates a forecast for a given time series using a Geometric Brownian Motion
    Based on code from JP Cianci @ Protocol Labs
    https://github.com/protocol/filecoin-agent-twin/blob/JP/agentfil/price.py

    Parameters
    ----------
    x : array-like
        The time series to forecast
    forecast_length : int
        The number of steps to forecast
    dt : float, optional
        The time step between observations
    seed : int, optional
        The seed for the random number generator
    """
    def _compute_params(x, dt=1):
        lp = np.log(x)
        r = np.diff(lp)

        sigma = np.std(r)/(dt**0.5)
        mu = np.mean(r)/dt + 0.5*sigma**2
        return mu, sigma

    drift, vol = _compute_params(x, dt)
    rng = np.random.RandomState(seed)
    
    prev_val = x[-1]
    y = np.zeros(forecast_length)
    for ii in range(forecast_length):
        log_update = (drift - 0.5 * vol**2) * dt + vol * dt**0.5 * rng.randn()
        y[ii] = prev_val * np.exp(log_update)
        prev_val = y[ii]

    return y