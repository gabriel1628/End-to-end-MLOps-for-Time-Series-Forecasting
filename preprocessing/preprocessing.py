from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean


def preprocessing_1(ts, forecast_horizon=24, n_lags=48, inference=False):
    ts = ts.sort_values(["unique_id", "ds"]).ffill()
    if inference == True:  # add rows between the last recorded value and the target
        unique_ids = ts["unique_id"].unique()
        for unique_id in unique_ids:
            new_rows = pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": pd.date_range(
                        ts["ds"].iloc[-1], periods=forecast_horizon, freq="h"
                    ),
                    "y": -99,  # must not be None
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

    X = fcst.preprocess(ts)
    if inference == False:
        X, y = X.iloc[:, 3:], X["y"]
        return X, y
    else:
        return X.iloc[:, 3:]


def preprocessing_2(ts, forecast_horizon=24, n_lags=48, inference=False):
    ts = ts.sort_values(["unique_id", "ds"]).ffill()
    if inference == True:  # add rows between the last recorded value and the target
        unique_ids = ts["unique_id"].unique()
        for unique_id in unique_ids:
            new_rows = pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": pd.date_range(
                        ts["ds"].iloc[-1], periods=forecast_horizon, freq="h"
                    ),
                    "y": -99,  # must not be None
                }
            )
            ts = pd.concat((ts, new_rows))
    else:  # drop discontinuous time series
        ids_to_drop = [21, 26, 41, 44, 47, 68]
        ts = ts[~ts["unique_id"].isin(ids_to_drop)]

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

    X = fcst.preprocess(ts)
    if inference == False:
        X, y = X.iloc[:, 3:], X["y"]
        return X, y
    else:
        return X.iloc[:, 3:]
