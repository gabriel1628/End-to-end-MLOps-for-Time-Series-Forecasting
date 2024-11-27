from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean


def preprocessing_1(
    ts,
    id_col="prediction_unit_id",
    time_col="datetime",
    target_col="consumption",
    forecast_horizon=48,
    n_lags=48,
    inference=False,
):
    ts = ts.sort_values([id_col, time_col]).ffill()
    if inference == True:  # add rows between the last recorded value and the target_col
        unique_ids = ts[id_col].unique()
        for id in unique_ids:
            new_rows = pd.DataFrame(
                {
                    id_col: id,
                    time_col: pd.date_range(
                        ts[time_col].iloc[-1], periods=forecast_horizon, freq="h"
                    ),
                    target_col: -99,  # must not be None
                }
            )
            ts = pd.concat((ts, new_rows))

    fcst = MLForecast(
        models=[],
        freq="h",
        lags=[i + forecast_horizon for i in range(n_lags)],
        lag_transforms={
            1: [ExpandingMean()],
            1: [RollingMean(window_size=24)],
            24: [RollingMean(window_size=24)],
        },
        date_features=["month", "dayofweek", "hour"],
    )

    X = fcst.preprocess(ts, id_col=id_col, time_col=time_col, target_col=target_col)
    if inference == False:
        X, y = X.iloc[:, 3:], X[target_col]
        return X, y
    else:
        return X.iloc[:, 3:]


def preprocessing_2(
    ts,
    id_col="prediction_unit_id",
    time_col="datetime",
    target_col="consumption",
    forecast_horizon=48,
    n_lags=48,
    inference=False,
):
    ts = ts.sort_values([id_col, time_col]).ffill()
    if inference == True:  # add rows between the last recorded value and the target_col
        unique_ids = ts[id_col].unique()
        for id in unique_ids:
            new_rows = pd.DataFrame(
                {
                    id_col: id,
                    time_col: pd.date_range(
                        ts[time_col].iloc[-1], periods=forecast_horizon, freq="h"
                    ),
                    target_col: -99,  # must not be None
                }
            )
            ts = pd.concat((ts, new_rows))
    else:  # drop discontinuous time series
        ids_to_drop = [21, 26, 41, 44, 47, 68]
        ts = ts[~ts[id_col].isin(ids_to_drop)]

    fcst = MLForecast(
        models=[],
        freq="h",
        lags=[i + forecast_horizon for i in range(n_lags)],
        lag_transforms={
            i + forecast_horizon: [ExpandingMean(), RollingMean(window_size=24)]
            for i in range(24)
        },
        date_features=["month", "dayofweek", "hour"],
    )

    X = fcst.preprocess(ts, id_col=id_col, time_col=time_col, target_col=target_col)
    if inference == False:
        X, y = X.iloc[:, 3:], X[target_col]
        return X, y
    else:
        return X.iloc[:, 3:]
