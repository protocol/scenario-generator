import datetime
import jax.numpy as jnp

# TODO: path append can be removed when mechafil is converted to a python module
import sys
sys.path.append('../filecoin-mecha-twin')
from mechafil.data import query_starboard_daily_power_onboarded


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
    rb_onboard_vec = onboards_df['day_onboarded_rb_power_pib']
    return t_vec, rb_onboard_vec