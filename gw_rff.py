import numpy as np
import pandas as pd
import time
from os.path import exists
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from IPython.display import display
import matplotlib.pyplot as plt
import os
import xarray as xr
import copy

USE_CUDA = False
USE_TYPE = np.float32

if USE_CUDA:
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
    import numpy as np
    import skcuda.linalg as linalg

    linalg.init()


def R2(y, y_hat):
    return 1 - (y - y_hat).var() / y.var()


def corr0(y, y_hat):
    return (y.T*y_hat).mean()/y.std()/y_hat.std()


class GW_Loader():
    data_dir = "C:\\Dev\\DataFree\\"
    defs = pd.DataFrame.e.read_string("""
name	target	formula	avail
Index			Index
Dividends past 12m			D12
Earnings past 12m			E12
Book to Market Ratio	b/m	b/m	
Net Equity Expansion	ntis	ntis	
Dividend Payout Ratio	de	D12/E12	
Dividend Price Ratio	dp	D12/Index	
Dividend Yield	dy	D12/Index_12	
Earnings Price Ratio	ep	E12/Index	
Stock Variance	svar	svar	
Cross-Sectional Premium		csp	
CRSP_SPvw			CRSP_SPvw
CRSP_SPvwx			CRSP_SPvwx
Risk Free			Rfree
Inflation	infl	infl	
Long Term Rate of Return	ltr	ltr	
T-Bills	tbl	tbl	
Long Term Yield	lty	lty	
Term Spread	tms	lty-tbl	
Corp Bond Return	corpr	corpr	AAA
AAA Yield			BAA
BAA Yield			
Default Return Spread	dfr	ltr-corpr	
Default Yield Spread	dfy	BAA-AAA	
Return 1M	ret_1m	Index/Index_1	
Return 12M	ret_12m	Index/Index_12	
Return 1M Fwd	ret_1m_fwd	Index__1/Index	
    """)

    def load_gw(self):
        df = pd.read_excel(self.data_dir + "Welsh_Goyal_PredictorData2021.xlsx", engine='openpyxl', sheet_name="Monthly")
        df["date"] = pd.to_datetime(df.yyyymm.astype(str)+"15") + pd.tseries.offsets.MonthEnd(0)
        df = df.drop(columns=["yyyymm"])
        df = df.set_index("date")

        df0 = pd.read_pickle("..\\macro_lens\\all.pkl")
        df = pd.concat([df, df0["sbbi_stocks"]], axis=1)

        df.columns.name = "predictor"
        df.index.name = "date"
        return df

    def calc_fields(self, df):
        """
        Data page 37
        * Our forecast target is the monthly return of the CRSP value-weighted index.

        * The information set we use for prediction consists of the 15 predictor variables from Welch and Goyal (2008)
          that are available at the monthly frequency over the sample 1926.
        * dfy, in, svar, de, lty, tms, tbl, dfr, dp, dy, ltr, ep, b/m, ntis, and ret_1m
        * We volatility standardize returns and predictors using backward-looking standard deviations that preserve the
          out-of-sample nature of our forecasts.
        * Returns are standardized by their trailing 12-month return standard deviation (to capture their comparatively
          fast moving conditional volatility),
        * while predictors are standardized using an expanding window historical standard deviation (given the much
          higher persistence of most predictors).
        * We require 36 months of data to ensure that we have enough stability in our initial predictor standardization,
          so the sample that we bring to our analysis begins in 1930.
        * We perform this standardization to align the empirical analysis with our homoskedastic theoretical
        """
        # Dividend Payout Ratio
        df["de"] = np.log(df["D12"] / df["E12"])

        # Dividend Price Ratio
        df["dp"] = np.log(df["D12"] / df["Index"])

        # Dividend Yield
        df["dy"] = np.log(df["D12"] / df["Index"].shift(12))

        # Earnings Price Ratio
        df["ep"] = np.log(df["E12"] / df["Index"])

        # Return 1M
        if True:
            df["ret_1m"] = df["sbbi_stocks"] - df["Rfree"]
        else:
            df["ret_1m"] = df["Index"].pct_change() - df["Rfree"]

        if True:
            df["ret_1m"] = np.log(1 + df.ret_1m)

        df["ret_12m"] = df["ret_1m"].rolling(12).sum()
        df["ret_1m_fwd"] = df["ret_1m"].shift(-1)

        # Term Spread
        df["tms"] = df["lty"] - df["tbl"]

        # Default Return Spread
        df["dfr"] = df["ltr"] - df["corpr"]

        # Default Yield Spread
        df["dfy"] = df["BAA"] - df["AAA"]

        return df

    def get_raw(self):
        df = self.load_gw()
        df = self.calc_fields(df)

        df = df[self.defs.target.dropna()]
        df = df[slice("1927", "2020")]

        return df

    def normalise(self, df):
        # all except returns normalised via growing window
        ret_col = df.columns.str.startswith("ret")
        non_ret = df.columns[~ret_col]
        ret_col = df.columns[ret_col]

        df_norm = df.copy(True)
        df_norm[non_ret] = df[non_ret] / df[non_ret].rolling(10000, min_periods=36).std()

        df_norm["vol_12m"] = df["ret_1m"].rolling(12).std()

        # use this to normalise also forward returns
        df_norm["ret_1m"] = df["ret_1m"] / df_norm.vol_12m
        df_norm["ret_12m"] = df["ret_12m"] / df_norm.vol_12m / np.sqrt(12)
        df_norm["ret_1m_fwd"] = df["ret_1m_fwd"] / df_norm.vol_12m

        df_norm = df_norm[slice("1930", "2020")]
        df_norm.to_pickle("df_norm.pkl")
        df_norm.to_csv("df_norm.csv")

        return df_norm

    def plot(self, df):
        df.notnull().sum(axis=1).plot()
        plt.show()
        df.stack().to_xarray().plot(x="date", col="predictor", col_wrap=3, sharey=False)
        plt.show()
        display(df.corr().e.style_corr())


def sin_cos(x):
    S = np.concatenate([
        np.sin(x)[:, :x.shape[1]//2, None],
        np.cos(x)[:, :x.shape[1]//2, None]
    ], axis=2)

    S = S.transpose((0, 2, 1))
    S = S.reshape([S.shape[0], S.shape[1] * S.shape[2]])

    return S


class GW_RFF(object):
    Ns_exp = list(np.arange(2, 20)) + list(np.exp(np.linspace(np.log(20), np.log(12000), 82)).round().astype(int))
    Ns_round = (
            list(np.arange(2, 30)) +
            list(np.arange(30, 50, 5)) +
            list(np.arange(50, 100, 10)) +
            list(np.arange(100, 500, 50)) +
            list(np.arange(500, 1000, 100)) +
            list(np.arange(1000, 5000, 500)) +
            list(np.arange(5000, 12001, 1000))
    )
    # self.Ns_coarse = [2, 6, 9, 11, 12, 13, 15, 150, 1500, 12000]
    Ns_coarse = [6, 12, 50, 120, 200, 300, 400, 500, 600, 1200, 5000, 12000]
    zs_3 = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    zs_6 = [10**i for i in range(-6, 7)]

    feature_names_all = [
            "b/m", "de", "dfr", "dfy", "dp", "dy", "ep", "infl",
            "ltr", "lty", "ntis", "svar", "tbl", "tms", "ret_1m",
            "corpr", "ret_12m",
    ]
    def to_small_dict(self):
        a = vars(self)
        res = {}
        for k, v in a.items():
            if hasattr(v, "size"):
                if v.size > 100:
                    continue
            res[k] = v

        return res

    def __init__(self, **kwargs):
        # global settings
        self.T = 12
        self.c_max = 1000
        self.gamma = 1
        self.kernel = np.sin
        self.c_ensembles = 50
        self.zs = self.zs_3

        self.feature_names = [
            "b/m", "de", "dfr", "dfy", "dp", "dy", "ep", "infl",
            "ltr", "lty", "ntis", "svar", "tbl", "tms", "ret_1m",
            #"corpr", "ret_12m",
        ]
        self.target_name = 'ret_1m_fwd'
        self.out_dir = "res"
        self.Ns = "Ns_round"
        self.seed = 0

        for key, value in kwargs.items():
            setattr(self, key, value)

        if isinstance(getattr(self, "Ns"), str):
            # only if self.Ns is a string, i.e. pointing to Ns_coarse, Ns_exp, Ns_round
            self.Ns = getattr(self, getattr(self, "Ns"))

        self.N_max = self.c_max * self.T
        self.ensembles = np.arange(self.c_ensembles)
        self.c_predictors = len(self.feature_names)

        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)

        if os.path.isfile(self.out_dir + "\\def.csv"):
            display(pd.read_csv(self.out_dir + "\\def.csv").set_index('Unnamed: 0')["0"].to_dict())
        else:
            pd.Series(self.to_small_dict()).to_csv(self.out_dir + "\\def.csv")


    def init_rff(self):
        # load normed features
        df_norm = pd.read_csv("df_norm.csv")
        df_norm["date"] = pd.to_datetime(df_norm.date)
        df_norm = df_norm.set_index("date").astype(np.float64)

        x = df_norm[self.feature_names].values
        y = df_norm[[self.target_name]].values

        np.random.seed(self.seed)
        w = np.random.randn(self.c_predictors, self.N_max) * self.gamma

        S = self.kernel(x @ w)

        if not os.path.isfile(f".\\{self.out_dir}\\w.pkl"):
            pd.DataFrame(w).to_pickle(f".\\{self.out_dir}\\w.pkl")
        if not os.path.isfile(f".\\{self.out_dir}\\df_norm.csv"):
            df_norm.to_csv(f".\\{self.out_dir}\\df_norm.csv")

        self.ts = np.arange(self.T, y.shape[0])

        if False:
            p = np.random.permutation(x.shape[0])
            S_rand = np.repeat(S[:, :, None], x.shape[1], axis=2)
            for c in range(x.shape[1]):
                x_ = x.copy()
                x_[:, c] = x_[p, c]
                S_rand[:, :, c] = kernel(x_ @ w)
        else:
            S_rand = None

        self.S = S
        self.y = y
        self.w = w
        self.x = x
        self.df_norm = df_norm
        self.S_rand = S_rand

        return self

    def fit_one(self, t, ensemble, z, N, use="mix", **kwargs):
        if use == "mix":
            if N > 50:
                use = "sk"
            else:
                use ="np"

        if (ensemble + 1) * N > self.N_max:
            return (None, None, None, None, None, None, None)

        start_time = time.time()
        # data
        S_train = self.S[t - self.T:t, N * ensemble:N * ensemble + N]
        S_test = self.S[t:t+1, N * ensemble:N * ensemble + N]
        y_train = self.y[t - self.T:t, :]
        y_test = self.y[t]
        #S_test_rand = S_rand[t, N * ensemble:N * ensemble + N, :].T

        S_train = S_train.astype(USE_TYPE)
        S_test = S_test.astype(USE_TYPE)
        y_train = y_train.astype(USE_TYPE)
        y_test = y_test.astype(USE_TYPE)
        #S_test_rand = S_test_rand.astype(USE_TYPE)

        if use == "cuda":
            # to cuda
            s_gpu = gpuarray.to_gpu(S_train)
            y_gpu = gpuarray.to_gpu(y_train)

            # model
            a_gpu = linalg.eye(N, USE_TYPE)
            a_gpu = linalg.add_dot(
                s_gpu, s_gpu, a_gpu,
                'T', 'N',
                1/self.T, z
            )
            b_gpu = linalg.dot(
                s_gpu, y_gpu,
                'T', 'N',
            )
            linalg.scale(1 / self.T, b_gpu)

            linalg.cho_solve(a_gpu, b_gpu)
            beta = b_gpu.get()

        elif use == "np":
            # model
            A = self.T * z * np.identity(N) + S_train.T @ S_train
            B = S_train.T @ y_train

            beta = np.linalg.solve(A, B)

        elif use == "sk":
            beta = Ridge(alpha=z*self.T, fit_intercept=False, **kwargs)
            beta = beta.fit(S_train, y_train).coef_.T

        # prediction on CPU
        y_test_hat = beta.T @ S_test.astype(USE_TYPE).T
        y_test_hat = y_test_hat.astype(USE_TYPE)[0, 0]
        beta_norm = np.linalg.norm(beta)

        #y_test_rand = beta.T @ S_test_rand.astype(USE_TYPE).T

        return (
            t,
            z,
            N,
            ensemble,
            y_test_hat,
            y_test[0],
            beta_norm,
            time.time() - start_time,
            #y_test_rand,
            #beta
        )

    res_col = [
        "t",
        "z",
        "N",
        "ensemble",
        "y_test_hat",
        "y_test",
        "beta_norm",
        "time",
        #"y_test_hat_rand",
        #"beta",
    ]

    def fit(self, z=None, N=None, use="mix"):
        if N is None:
            for N in self.Ns:
                _ = self.fit(z=z, N=N, use=use)
                print(N, end="\r")
            return self

        if z is None:
            for z in self.zs:
                _ = self.fit(z=z, N=N, use=use)
            return self

        start_time = time.time()
        file = f".\\{self.out_dir}\\res_{int(np.log10(z))}_{N}.pkl"

        if exists(file):
            return

        # create dummy file so no other process starts the task
        pd.DataFrame().to_pickle(file)

        rs = []
        for t in self.ts:
            for ensemble in self.ensembles:
                r = self.fit_one(t, ensemble, z, N, use=use)
                rs += [r]

            print(t, end="\r")

        res = pd.DataFrame(rs, columns=self.res_col).dropna()

        print(z, N, time.time()-start_time)
        res.to_pickle(file)

        return res

    def get_tx(self):
        # and xr version
        rex_ts = self.res_ts.set_index(["z_log", "N", "ensemble", "t"])[
            ["ret", "mkt", "weight", "beta_norm"]].to_xarray()
        rex_ts = rex_ts.astype(float)
        rex_ts = rex_ts.assign_coords(t=self.df_norm.index[rex_ts.t.astype(int)].values)
        self.rex_ts = rex_ts
        return rex_ts

    def get_ts(self, z=None, N=None):
        if N is None and z is None:
            res_ts = []
            for N in self.Ns:
                for z in self.zs:
                    try:
                        res_ts += [self.get_ts(z=z, N=N)]
                    except:
                        pass
                print(N, end="\r")
            self.res_ts = pd.concat(res_ts)
            self.res_ts.to_pickle(self.out_dir + "\\res_ts.pkl")

            return self.res_ts

        file = f".\\{self.out_dir}\\res_{int(np.log10(z))}_{N}.pkl"

        res_ts = pd.read_pickle(file)

        res_ts["z_log"] = np.log10(res_ts.z).astype(int)
        res_ts["c"] = res_ts.N / self.T
        res_ts["mkt"] = res_ts.y_test
        res_ts["ret"] = res_ts.y_test_hat * res_ts.y_test
        res_ts["weight"] = res_ts.y_test_hat
        res_ts["N"] = res_ts.N.astype(int)

        return res_ts

    def get_stats(self, res=None, y_test_hat="y_test_hat"):
        if res is None:
            res_group = self.res_ts.groupby(["N", "z", "ensemble"])

            res_stats = []
            for g, res in res_group:
                res_stats += [self.get_stats(res, y_test_hat=y_test_hat)]
                print(g, end="\r")

            self.res_stats = pd.concat(res_stats, axis=1).T

            self.res_stats.to_pickle(self.out_dir + "\\res_stats.pkl")

            self.rex_stats = self.res_stats.set_index(["z_log", "N", "ensemble"]).drop(
                columns="y_test_hat").to_xarray().to_array("stat")
            return self.res_stats

        if False:
            res = res.copy(True)
            res["y_test_hat"] = res[y_test_hat]

        res["weight"] = res["y_test_hat"]
        res["mkt"] = res.y_test
        res["ret"] = res.weight * res.mkt

        capm = sm.OLS(res.ret, sm.add_constant(res.mkt)).fit()

        z = res.z.iloc[0]
        N = res.N.iloc[0]
        ensemble = res.ensemble.iloc[0]

        res_stats = dict(
            z=z,
            z_log=int(np.log10(z)),
            ensemble=ensemble,
            y_test_hat=y_test_hat,
            N=N,
            c=N / self.T,
            net_exp=res.weight.mean(),
            gross_exp=res.weight.abs().mean(),
            R2=R2(res.y_test, res.y_test_hat),
            corr0=corr0(res.y_test, res.y_test_hat),
            beta_norm=res.beta_norm.mean(),
            Ret=res.ret.mean() * 12,
            Vol=res.ret.std() * np.sqrt(12),
            Sharpe=res.ret.mean() / res.ret.std() * np.sqrt(12),
            Alpha=capm.params["const"],
            Beta=capm.params["mkt"],
            IR=capm.params["const"] / capm.resid.std() * np.sqrt(12),
            Alpha_t=capm.tvalues["const"],
        )

        res_stats = pd.Series(res_stats)
        return res_stats

    def get_statx(self, denormalise=False):
        rex_ts = self.rex_ts.copy(True)

        if denormalise:
            rex_ts["vol_12m"] = self.df_norm.rename_axis(index="t").vol_12m[rex_ts.t.values]
            rex_ts["mkt"] = rex_ts.mkt * rex_ts.vol_12m
            rex_ts["weight"] = rex_ts.weight  # * rex_ts.vol_12m
            rex_ts["ret"] = rex_ts.weight * rex_ts.mkt

        rex_ts["ret_10"] = rex_ts.ret / rex_ts.ret.std("t") * 0.1 / np.sqrt(12)

        rex_stats = xr.Dataset()
        rex_stats["Ret"] = rex_ts.ret.mean("t") * 12
        rex_stats["Ret_10"] = rex_ts.ret_10.mean("t") * 12
        rex_stats["Vol"] = rex_ts.ret.std("t") * np.sqrt(12)
        rex_stats["Sharpe"] = rex_ts.ret.mean("t") / rex_ts.ret.std("t") * np.sqrt(12)

        rex_stats["Net_exp"] = rex_ts.weight.mean("t")
        rex_stats["Gross_exp"] = rex_ts.weight.pipe(np.abs).mean("t")
        self.rex_stats = rex_stats
        return self.rex_stats


class GW_RFFs(GW_RFF):

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        super().__init__(**kwargs)

    def fit(self, rep=None):
        if rep is None:
            # run all
            for rep in self.ensembles:
                self.fit(rep)

            return self

        out_dir = self.out_dir + f"\\rep_{rep}"
        if os.path.isdir(out_dir):
            return

        init_kwargs = {
            **self.init_kwargs,
            **dict(out_dir=out_dir, c_ensembles=1, seed=rep)
        }

        rff = GW_RFF(**init_kwargs)

        rff.init_rff()
        rff.fit()
        rff.get_ts()
        rff.res_ts.to_pickle(out_dir + "\\res_ts.pkl")

        return self

    def get_ts(self):
        try:
            self.res_ts = pd.read_pickle(self.out_dir + "\\res_ts.pkl")
            return self.res_ts
        except:

            res_ts = []

            for rep in self.ensembles:
                out_dir = self.out_dir + f"\\rep_{rep}"
                try:
                    df = pd.read_pickle(out_dir + "\\res_ts.pkl")
                    df["ensemble"] = rep
                    res_ts += [df]
                    print(rep, end="\r")
                except:
                    pass

            self.res_ts = pd.concat(res_ts)
            self.res_ts.to_pickle(self.out_dir + "\\res_ts.pkl")
            return self.res_ts


if __name__ == "__main__":
    pass
