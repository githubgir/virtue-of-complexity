import pandas as pd
import numpy as np
import xarray as xr
from IPython.core.display import display

@pd.api.extensions.register_dataframe_accessor("e")
@pd.api.extensions.register_series_accessor("e")
class PDE:
    def __init__(self, obj):
        self._obj = obj

    def corr_0(self):
        df = self._obj
        c = (df.values[:, :, None] * df.values[:, None, :])
        c = np.nansum(c, axis=0)
        v = 1 / np.diag(c) ** 0.5
        c = v[:, None] * c * v[None, :]
        c = pd.DataFrame(c, index=df.columns, columns=df.columns)
        return c

    def perf_stats(self, af=None):
        obj = self._obj

        if af is None:
            d = obj.index.to_frame().iloc[:, 0]
            d = d.diff() / np.timedelta64(1, "D")
            d = d.median()
            if d == 1:
                af = 252
            elif d == 7:
                af = 52
            elif d >= 25 and d <= 31:
                af = 12

        if isinstance(obj, pd.DataFrame):
            cols = obj.columns
            res = [obj[c].e.perf_stats() for c in cols]
            return pd.concat(res, keys=cols, axis=1).T

        stats = {
            "Avg Return": obj.mean() * af,
            "Period Return": (obj + 1).prod(),
            "Volatility": obj.std() * np.sqrt(af),
            "Sharpe": obj.mean() / obj.std() * np.sqrt(af),
        }
        stats = pd.Series(stats)
        return stats

    def style_corr(self):
        df = self._obj
        return df.style.background_gradient(axis=None, cmap="seismic", vmin=-1, vmax=1).format(precision=2)

    def plot_cum_ret(self):
        o = self._obj

        display(o.e.perf_stats())
        display(o.corr())
        return o.cumsum().plot()

    def normalise(self, mean=0, vol=1):
        x = self._obj
        if mean is not None:
            x = x - x.mean() + mean
        if vol is not None:
            x = x / x.std() * vol

        return x

    @staticmethod
    def read_string(df_as_string):
        from io import StringIO
        return pd.read_csv(StringIO(df_as_string), sep="\t")


@xr.register_dataset_accessor("e")
@xr.register_dataarray_accessor("e")
class XRE:
    def __init__(self, obj):
        self._obj = obj

    def plot_cum_ret(self):
        o = self._obj
        return o.cumsum(o.dims[0]).plot(x=o.dims[0], hue=o.dims[1])

    def perf_stats(self, af=None):
        obj = self._obj

        df = obj.to_series().unstack(obj.dims[0]).T
        stats = df.e.perf_stats(af=af)

        return stats