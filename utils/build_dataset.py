import pandas as pd
import numpy as np
from scipy.special import roots_gegenbauer
import requests
from scipy.special import eval_gegenbauer
import datetime
import argparse

def add_swap_size_metrics():
    df_integral = pd.read_parquet('./data/Uniswap/swap_size_metrics.parquet')
    return df_integral


def add_forecasting_target():
    state = pd.read_parquet('./data/Uniswap/hourly_pool_state_full.parquet')
    state.index = state.hour
    return state['poolTick'].astype(float)

def load_uniswap_metrics():
    metrics1 = pd.read_parquet('./data/Uniswap/USDC_USDT_hourly_metrics.parquet').query('feeTier == 100').sort_values(by='datetime').iloc[:-1,:]
    metrics5 = pd.read_parquet('./data/Uniswap/USDC_USDT_hourly_metrics.parquet').query('feeTier == 500').sort_values(by='datetime').iloc[:-1,:]

    metrics1.index = metrics1['datetime']
    metrics5.index = metrics5['datetime']
    metrics1 = metrics1[['tvlUSD', 'net_amountUSD', 'net_amount0', 'swap_count']]
    metrics5 = metrics5[['swap_count', 'net_amountUSD', 'tvlUSD']].ffill().bfill()
    try:
        metrics = metrics1.join(metrics5, rsuffix='_500', lsuffix='_100')
    except Exception as e:
        print(e)
        print('--- could not join uniswap metrics for 100 and 500 fee tier ---')
        print('100 and 500 feeTier data last dates :', metrics1.index[-1], metrics5.index[-1])

    
    return metrics

def full_aave(coin = 'usdt'):
    aave_v2 = pd.read_parquet(f'./data/AAVE/aave_v2_{coin}_eth.parquet').query('index >= "2022"')
    aave_v3 = pd.read_parquet(f'./data/AAVE/aave_v3_{coin}_eth.parquet').query('index >= "2022"')
    aave_v2['utilisation_rate'] = aave_v2['borrowed_USD'] / aave_v2['supplied_USD']
    v2 = aave_v2[['supplied_USD', 'borrowed_USD', 'utilisation_rate']]

    aave_v3['utilisation_rate'] = aave_v3['borrowed_USD'] / aave_v3['supplied_USD']
    v3 = aave_v3[['supplied_USD', 'borrowed_USD', 'utilisation_rate']]
    df = pd.merge(v2,v3, how='outer', left_index=True, right_index=True, suffixes=('_v2','_v3'))
    tvl_v2 = df["supplied_USD_v2"].fillna(0)
    tvl_v3 = df["supplied_USD_v3"].fillna(0)

    use_v3 = tvl_v3 > tvl_v2

    out = pd.DataFrame(index=df.index)
    # out["version"] = np.where(use_v3, "v3", "v2")

    out[f"supplied_USD_{coin}"] = np.where(use_v3,
                                df["supplied_USD_v3"].fillna(0),
                                df["supplied_USD_v2"].fillna(0))

    out[f"utilisation_rate_{coin}"] = np.where(use_v3,
                                   df["utilisation_rate_v3"].fillna(0),
                                   df["utilisation_rate_v2"].fillna(0))
    return out

def full_aave_metrics():
    out_usdt = full_aave('usdt').iloc[:-1,:]
    out_usdc = full_aave('usdc').iloc[:-1,:]
    out_usdt.index = out_usdt.index.tz_localize('UTC')
    out_usdc.index = out_usdc.index.tz_localize('UTC')
    aave_metrics = pd.concat([out_usdt, out_usdc], axis=1)
    return aave_metrics

def aave_liquidations():
    liquidations_v3 = pd.read_parquet('./data/AAVE/liquidations/aave_v3_eth_liquidations_hourly.parquet')
    liquidations_v2 = pd.read_parquet('./data/AAVE/liquidations/aave_v2_eth_liquidations_hourly.parquet')

    liquidations = liquidations_v2.join(liquidations_v3, how = 'outer', lsuffix = '_v2', rsuffix = '_v3').fillna(0)
    liquidations = liquidations[liquidations.index >= datetime.datetime(2022,1,1, tzinfo=datetime.timezone.utc)]
    liquidations['liquidation_USD'] = liquidations['liquidation_usd_v2'] + liquidations['liquidation_usd_v3']
    return liquidations[['liquidation_USD']]

def crv_3pool_metrics():
    crv = pd.read_parquet('./data/Curve/curve_3pool_hourly.parquet').query('index >= "2022"')
    crv.index = crv.index.tz_localize('UTC')
    crv = crv.iloc[:-1,:]
    crv['curve_entropy'] = -(crv['w_USDC'] * np.log(crv['w_USDC']) + crv['w_USDT'] * np.log(crv['w_USDT']) + crv['w_DAI'] * np.log(crv['w_DAI']))
    crv_metrics = crv[['totalValueLockedUSD', 'hourlyVolumeUSD', 'w_USDC', 'w_USDT', 'curve_entropy']]
    return crv_metrics

def gegenbauer_timeseries_features(df, alpha=0.5, win = 24):
    df["E_shape_logret"] = df['E_shape'].pct_change()
    df["E_odd_logret"]  = df['E_odd'].pct_change()
    df["E_even_logret"] = df['E_even'].pct_change()

    df["odd_ratio_logret"]  = df['odd_ratio'].pct_change()
    df["even_ratio_logret"] = df['even_ratio'].pct_change()
    df["odd_to_even_ratio_logret"] = df['odd_to_even_ratio'].pct_change()

    df["E_low_logret"]  = df['E_low'].pct_change()
    df["E_mid_logret"]  = df['E_mid'].pct_change()
    df["E_high_logret"] = df['E_high'].pct_change()

    df["low_ratio_logret"]  = df['low_ratio'].pct_change()
    df["mid_ratio_logret"]  = df['mid_ratio'].pct_change()
    df["high_ratio_logret"] = df['high_ratio'].pct_change()
     
    for i in range(8):
        df[f'Gegenbauer_{alpha}_deg{i}_MA{win}'] =  df[f'Gegenbauer_{alpha}_deg{i}'].rolling(window = win, min_periods = 1).mean()
        df[f'Gegenbauer_{alpha}_deg{i}_logret'] =  df[f'Gegenbauer_{alpha}_deg{i}'].pct_change()
        df[f'Gegenbauer_{alpha}_deg{i}_vol{win}'] =  df[f'Gegenbauer_{alpha}_deg{i}_logret'].rolling(window = win, min_periods = 1).std()
    df = df.iloc[win:]
    return df

def add_technical_indicators(
    df: pd.DataFrame,
    price_col: str = "price_usd",
    sma_windows=(50*24,),
    ema_windows=(200*24,),
    rsi_period: int = 24,
    atr_period: int = 24,
    vol_window: int = 24 * 7,          # 14 days of hourly bars
    periods_per_year: int = 24 * 365,  # for annualized volatility
    high_col: str  = None,
    low_col: str  = None,
    close_col: str  = None,
    prefix: str = "",
) -> pd.DataFrame:
    out = df.copy().sort_index()
    close = out[price_col].astype(float)

    # --- SMA / EMA ---
    for w in sma_windows:
        out[f"{prefix}SMA_{w}"] = close.rolling(window=w, min_periods=w).mean()

    for w in ema_windows:
        out[f"{prefix}EMA_{w}"] = close.ewm(span=w, adjust=False, min_periods=w).mean()

    # --- RSI (Wilder) ---
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out[f"{prefix}RSI_{rsi_period}"] = 100 - (100 / (1 + rs))
    if high_col and low_col:
        high = out[high_col].astype(float)
        low = out[low_col].astype(float)
        c = out[close_col].astype(float) if close_col else close

        prev_close = c.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1
        ).max(axis=1)
    else:
        tr = close.diff().abs()

    out[f"{prefix}ATR_{atr_period}"] = tr.ewm(alpha=1 / atr_period, adjust=False, min_periods=atr_period).mean()
    log_ret = np.log(close).diff()
    out[f"{prefix}vol_{vol_window}"] = log_ret.rolling(window=vol_window, min_periods=vol_window).std()
    out[f"{prefix}vol_{vol_window}_ann"] = out[f"{prefix}vol_{vol_window}"] * np.sqrt(periods_per_year)
    # out[f'{prefix}price_raw'] = out['price_raw']
    out[f'{prefix}{price_col}'] = out[price_col]
    try:
        out.drop(columns=['price_raw', price_col], inplace=True)
    except:
        out.drop(columns=[price_col], inplace=True)
    return out

def eth_price_oracle():
    ETH_price = pd.read_parquet('./data/ETH_blocks/Chainlink/ethusd_oracle_hourly.parquet')
    ts = datetime.datetime(2022,1,1, tzinfo=datetime.timezone.utc)
    df = add_technical_indicators(ETH_price, price_col='price_usd', prefix = 'eth_')
    df= df[df.index >= ts]
    df = df.drop(columns = ['eth_vol_168_ann'])
    return df

def usd_index_oracle():
    USD_index = pd.read_parquet('./data/ETH_blocks/Chainlink/usd_index_hourly.parquet')
    ts = datetime.datetime(2022,1,1, tzinfo=datetime.timezone.utc)
    logdiff = np.log(USD_index['usd_index']).diff()
    df = USD_index.copy()
    df['fx_volatility'] = logdiff.rolling(window=24*7, min_periods=24*7).std()
    df= df[df.index >= ts]
    df = df.drop(columns = ['EUR/USD_fx_foreign_per_usd', 'GBP/USD_fx_foreign_per_usd', 'JPY/USD_fx_foreign_per_usd'])
    return df

def btc_price_oracle():
    BTC_price = pd.read_parquet('./data/ETH_blocks/Chainlink/btcusd_oracle_hourly.parquet')
    ts = datetime.datetime(2022,1,1, tzinfo=datetime.timezone.utc)
    df = add_technical_indicators(BTC_price, price_col='price_usd', prefix = 'btc_')
    df= df[df.index >= ts]
    df = df.drop(columns = ['btc_vol_168_ann'])
    return df

def fear_greed_index():
    r = requests.get(url = 'https://api.alternative.me/fng/', params = {'limit': 0, 'date_format': ''})
    r = r.json()['data']
    dfG = pd.DataFrame(r)
    dfG =dfG.iloc[::-1]

    dfG.index = [datetime.datetime.fromtimestamp(int(x)) for x in dfG.timestamp]
    dfG['value'] = dfG['value'].astype(float)
    dfG = dfG.resample('1h')['value'].mean().ffill()
    dfG.index = dfG.index.tz_localize('UTC')
    return dfG

def preprocess_liq_curve(df, TT = datetime.datetime(2022,1,1,0, tzinfo=datetime.timezone.utc)):
    d = df[df.hour >= TT]
    # d = d[d.hour <= TT_end]
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, unit = 's')
    WINDOW = 50
    d["tick_norm"] = (d["tickLower"].astype(float) - 0.0) / WINDOW
    d["tick_norm"] = d["tick_norm"].clip(-1, 1)

    # Log liquidity (use log1p to handle zeros)
    d["logL"] = np.log1p(d["active_liquidity_L"].astype(float))

    LIQ_COL = "active_liquidity_L"            
    Y_pegcentered = (d.pivot_table(index="timestamp", columns="tick_norm", values=LIQ_COL, aggfunc="sum")
        .sort_index()
        .sort_index(axis=1))
    Y_pegcentered = Y_pegcentered.fillna(0.0)
    Ylog_pegcentered = np.log1p(Y_pegcentered)
    return Ylog_pegcentered

def gegenbauer_vander(x, deg, alpha):
    if alpha <= -0.5:
        raise ValueError("Require alpha > -0.5.")
    x = np.asarray(x, float)
    return np.column_stack([eval_gegenbauer(n, alpha, x) for n in range(deg + 1)])

def orthopoly_decompose(
    Ylog,                 # array-like (n_t, n_ticks) OR pd.DataFrame with tick columns
    x_norm,               # array-like (n_ticks,), must be in [-1, 1]
    deg=5,                # highest degree; number of factors = deg+1
    center_time=True,     # subtract m(x) across time (like PCA centering)
    ridge=1e-10,          # small ridge for numerical stability
    index=None, 
    alpha = 0.5,          # optional time index if Ylog is ndarray
):
    if isinstance(Ylog, pd.DataFrame):
        Y = Ylog.to_numpy(dtype=float)
        if index is None:
            index = Ylog.index
    else:
        Y = np.asarray(Ylog, dtype=float)
        if index is None:
            index = pd.RangeIndex(Y.shape[0])

    x = np.asarray(x_norm, dtype=float)
    if np.any(x < -1 - 1e-12) or np.any(x > 1 + 1e-12):
        raise ValueError("x_norm must lie in [-1,1].")

    n_t, n_ticks = Y.shape
    if x.shape[0] != n_ticks:
        raise ValueError("x_norm length must match n_ticks (number of columns in Ylog).")

    # --- mean curve and centering (matches the paper's 'centered data' assumption)
    if center_time:
        m = Y.mean(axis=0)
        X = Y - m
    else:
        m = np.zeros(n_ticks)
        X = Y

    
    Phi = gegenbauer_vander(x, deg, alpha)   # (n_ticks, deg+1)
    basis_name = f"Gegenbauer_{alpha}"

    # --- solve for coefficients B in least squares sense for all t:
    # X ≈ B Phi^T  ->  B = X Phi (Phi^T Phi)^{-1}
    G = Phi.T @ Phi
    G = G + ridge * np.eye(G.shape[0])
    Ginv = np.linalg.inv(G)

    B = (X @ Phi) @ Ginv               # (n_t, deg+1)
    Yhat = (B @ Phi.T) + m             # (n_t, n_ticks)
    R = Y - Yhat

    # package coefficients as DataFrame for convenient "score through time"
    cols = [f"{basis_name}_deg{j}" for j in range(deg + 1)]
    B_df = pd.DataFrame(B, index=index, columns=cols)

    return m, B_df, Phi, Yhat, R


def gegenbauer_scores(Ylog, x_norm, deg=5, alpha=0.5, center_time=True, ridge=1e-10):
    mG, B_gegen, Phi_gegen, Yhat_gegen, R_gegen = orthopoly_decompose(
        Ylog, x_norm, deg=deg, center_time=center_time, ridge=ridge, alpha=alpha
    )
    return B_gegen, Yhat_gegen, R_gegen, mG, Phi_gegen

def decomp_logL_curve(alpha):
    df = pd.read_parquet('./data/Uniswap/hourly_liquidity_full.parquet')
    Ylog_pegcentered = preprocess_liq_curve(df)
    gegen_scores, yhat,_,m,phi = gegenbauer_scores(Ylog_pegcentered,
                                            np.linspace(-1, 1, Ylog_pegcentered.shape[1]), 
                                            deg = 7, 
                                            alpha=alpha, 
                                            center_time= False
                                            )
    return gegen_scores

def gegenbauer_energy_features(df, alpha=0.4):
    PREFIX = f"Gegenbauer_{alpha}_deg"
    eps = 1e-12

    cols = [f"{PREFIX}{i}" for i in range(8)]
    sq = df[cols].pow(2)

    df["E_total"] = sq.sum(axis=1)


    shape_cols = [f"{PREFIX}{i}" for i in range(1, 8)]
    df["E_shape"] = sq[shape_cols].sum(axis=1)


    odd_cols  = [f"{PREFIX}{i}" for i in [1, 3, 5, 7]]
    even_cols = [f"{PREFIX}{i}" for i in [2, 4, 6]]  # exclude 0 on purpose

    df["E_odd"]  = sq[odd_cols].sum(axis=1)
    df["E_even"] = sq[even_cols].sum(axis=1)

    df["odd_ratio"]  = df["E_odd"]  / (df["E_shape"] + eps)
    df["even_ratio"] = df["E_even"] / (df["E_shape"] + eps)
    df["odd_to_even_ratio"] = df["E_odd"] / (df["E_even"] + eps)


    low_cols  = [f"{PREFIX}{i}" for i in [1, 2]]
    mid_cols  = [f"{PREFIX}{i}" for i in [3, 4]]
    high_cols = [f"{PREFIX}{i}" for i in [5, 6, 7]]

    df["E_low"]  = sq[low_cols].sum(axis=1)
    df["E_mid"]  = sq[mid_cols].sum(axis=1)
    df["E_high"] = sq[high_cols].sum(axis=1)

    df["low_ratio"]  = df["E_low"]  / (df["E_shape"] + eps)
    df["mid_ratio"]  = df["E_mid"]  / (df["E_shape"] + eps)
    df["high_ratio"] = df["E_high"] / (df["E_shape"] + eps)
    return df 

def build_dataset(
            dataset_path,
            alpha, aave, aave_liq, crv, eth_price, 
            eth_indicators, btc_price, btc_indicators, usd_index, usd_indicators, fear_greed, gegen, target, 
            target_window, target_threshold, depeg_side, dynamic_threshold,
            gegen_indicators, swap_size, 
            bypass = False,
            **kwargs):
        dataset = load_uniswap_metrics()
        if aave:
            try:
                dataset = dataset.join(full_aave_metrics())
            except Exception as e:
                print(e)
                print('--- could not join aave metrics ---')    
                print('aave metrics last date :', full_aave_metrics().index[-1])
        
        if aave_liq:
            try:    
                dataset = dataset.join(aave_liquidations())
            except Exception as e:
                print(e)
                print('--- could not join aave liquidations ---')    
                print('aave liquidations last date :', aave_liquidations().index[-1])
        
        if crv:
            try:
                dataset = dataset.join(crv_3pool_metrics())
            except Exception as e:      
                print(e)
                print('--- could not join curve 3pool metrics ---')    
                print('curve 3pool metrics last date :', crv_3pool_metrics().index[-1])
        if eth_price:
            try:
                dataset = dataset.join(eth_price_oracle()[['eth_price_usd']])
            except Exception as e:      
                print(e)
                print('--- could not join eth price oracle ---')    
                print('eth price oracle last date :', eth_price_oracle().index[-1])
        if eth_indicators: 
            try:
                dataset = dataset.join(eth_price_oracle().drop(columns=['eth_price_usd']))
            except Exception as e:      
                print(e)
                print('--- could not join eth price oracle ---')    
                print('eth price oracle last date :', eth_price_oracle().index[-1])
        if btc_price:
            try:
                dataset = dataset.join(btc_price_oracle()[['btc_price_usd']])
            except Exception as e:      
                print(e)
                print('--- could not join btc price oracle ---')    
                print('btc price oracle last date :', btc_price_oracle().index[-1])
        if btc_indicators:
            try:
                dataset = dataset.join(btc_price_oracle().drop(columns=['btc_price_usd']))
            except Exception as e:      
                print(e)
                print('--- could not join btc price oracle ---')    
                print('btc price oracle last date :', btc_price_oracle().index[-1])
        if usd_index:
            try:
                dataset = dataset.join(usd_index_oracle()['usd_index'])
            except Exception as e:      
                print(e)
                print('--- could not join usd index oracle ---')    
                print('usd index oracle last date :', usd_index_oracle().index[-1])
        if usd_indicators:
            try:
                dataset = dataset.join(usd_index_oracle().drop(columns=['usd_index']))
            except Exception as e:      
                print(e)
                print('--- could not join usd index oracle ---')    
                print('usd index oracle last date :', usd_index_oracle().index[-1])
        if fear_greed: 
            try:
                dfG = fear_greed_index()
                dfG = dfG.reindex_like(dataset, method='ffill')
                dataset = dataset.join(dfG.rename('fear_greed_index'))
            except Exception as e:      
                print(e)
                print('--- could not load fear and greed index ---')    
        if gegen:
            try:
                gegen_scores = decomp_logL_curve(alpha)
                dataset = dataset.join(gegen_scores)
                dataset = gegenbauer_energy_features(dataset, alpha=alpha)
                if gegen_indicators:
                    try:
                        dataset = gegenbauer_timeseries_features(dataset, alpha = alpha)
                    except Exception as e:      
                        print(e)
                        print('--- could not join gegenbauer liquidity curve time series features ---')   
            except Exception as e:      
                print(e)
                print('--- could not join gegenbauer liquidity curve scores ---')   
        if swap_size:
            try:
                swap_size_metrics = add_swap_size_metrics()
                dataset = dataset.join(swap_size_metrics)
            except Exception as e:      
                print(e)
                print('--- could not join uniswap swap size curve metrics ---')
        
        try:
            dataset = dataset.join(add_forecasting_target())
        except Exception as e:
            print('--- could not add uniswap active pool price (forecasting target) ---')
            raise e
            
        if target:
            w = int(target_window)        # hours
            thr = int(target_threshold)   # bps
            x = dataset["poolTick"].astype(float)      # depeg in bps (can be + or -)
            upper = x.rolling(30*24, min_periods=1).quantile(0.9975)
            lower = x.rolling(30*24, min_periods=1).quantile(0.0025)

            start = dataset.index.min()
            dyn_ok = pd.Series(
                dataset.index >= (start + pd.Timedelta(days=30)),
                index=dataset.index
            )
            
            x1     = x.shift(-1)
            upper1 = upper.shift(-1)
            lower1 = lower.shift(-1)

            dyn_ok1 = dyn_ok.shift(-1).infer_objects(copy=False)
            if depeg_side == "up":
                x1_hard = x1.clip(lower=0)
            elif depeg_side == "down":
                x1_hard = x1.clip(upper=0)
            else:
                x1_hard = x1
           
            future_max = x1_hard.iloc[::-1].rolling(w, min_periods=1).max().iloc[::-1]
            future_min = x1_hard.iloc[::-1].rolling(w, min_periods=1).min().iloc[::-1]
            hit_hard   = (future_max >= thr) | (future_min <= -thr)
          
            if depeg_side == "up":
                out_of_band = (x1 > upper1)
            elif depeg_side == "down":
                out_of_band = (x1 < lower1)
            else:
                out_of_band = (x1 > upper1) | (x1 < lower1)
            
            out_of_band = (out_of_band & dyn_ok1).fillna(False)

            hit_dyn = out_of_band.iloc[::-1].rolling(w, min_periods=1).max().iloc[::-1].astype(bool)
            if dynamic_threshold:
                dataset["target"] = (hit_hard | hit_dyn).fillna(False).astype(int)
            else:
                dataset["target"] = hit_hard.fillna(False).astype(int)

        dataset = dataset.astype('float32').ffill()
        if target:
            if not (aave and aave_liq and crv and eth_price and eth_indicators and btc_price and btc_indicators and fear_greed and gegen and gegen_indicators and swap_size and usd_index and usd_indicators): 
                if not bypass:
                    dataset.to_parquet(f'{dataset_path}/dataset_alpha_{alpha}_aave-{aave}_ethprice-{eth_price}_ethind-{eth_indicators}_btcprice-{btc_price}_btcind-{btc_indicators}_fear-{fear_greed}_gegen-{gegen}_gegenind-{gegen_indicators}_swap-{swap_size}_usdi-{usd_index}_usdiind-{usd_indicators}_binarytarget_win-{target_window}_thresh-{target_threshold}_{depeg_side}_dynamic-{dynamic_threshold}.parquet')
                return f'{dataset_path}/dataset_alpha_{alpha}_aave-{aave}_ethprice-{eth_price}_ethind-{eth_indicators}_btcprice-{btc_price}_btcind-{btc_indicators}_fear-{fear_greed}_gegen-{gegen}_gegenind-{gegen_indicators}_swap-{swap_size}_usdi-{usd_index}_usdiind-{usd_indicators}_binarytarget_win-{target_window}_thresh-{target_threshold}_{depeg_side}_dynamic-{dynamic_threshold}.parquet' 
            else:
                if not bypass:
                    dataset.to_parquet(f'{dataset_path}/dataset_alpha_{alpha}_full_binarytarget_win-{target_window}_thresh-{target_threshold}_{depeg_side}_dynamic-{dynamic_threshold}.parquet')
                return f'{dataset_path}/dataset_alpha_{alpha}_full_binarytarget_win-{target_window}_thresh-{target_threshold}_{depeg_side}_dynamic-{dynamic_threshold}.parquet'
        else:
            if not (aave and aave_liq and crv and eth_price and eth_indicators and btc_price and btc_indicators and fear_greed and gegen and gegen_indicators and swap_size and usd_index and usd_indicators):   
                if not bypass:
                    dataset.to_parquet(f'{dataset_path}/dataset_alpha_{alpha}_aave-{aave}_ethprice-{eth_price}_ethind-{eth_indicators}_btcprice-{btc_price}_btcind-{btc_indicators}_fear-{fear_greed}_gegen-{gegen}_gegenind-{gegen_indicators}_swap-{swap_size}_usdi-{usd_index}_usdiind-{usd_indicators}.parquet')
                return f'{dataset_path}/dataset_alpha_{alpha}_aave-{aave}_ethprice-{eth_price}_ethind-{eth_indicators}_btcprice-{btc_price}_btcind-{btc_indicators}_fear-{fear_greed}_gegen-{gegen}_gegenind-{gegen_indicators}_swap-{swap_size}_usdi-{usd_index}_usdiind-{usd_indicators}.parquet'
            else:
                if not bypass:
                    dataset.to_parquet(f'{dataset_path}/dataset_alpha_{alpha}_full.parquet')
                return f'{dataset_path}/dataset_alpha_{alpha}_full.parquet'
         
def add_dataset_args(parser):
    dataset_building = parser.add_argument_group('dataset building arguments')
    dataset_building.add_argument('--dataset_path', type=str, default='./preprocessed_datasets', help='path to save the dataset')
    dataset_building.add_argument('--alpha', type=float, help='Gegenbauer polynomial alpha parameter', required=True)
    dataset_building.add_argument('--aave',action='store_false', help='remove AAVE metrics')
    dataset_building.add_argument('--aave_liq',action='store_false', help='remove AAVE liquidations')
    dataset_building.add_argument('--crv',action='store_false', help='remove Curve 3pool metrics')
    dataset_building.add_argument('--eth_price',action='store_false', help='remove ETH price oracle')
    dataset_building.add_argument('--eth_indicators',action='store_false', help='remove ETH price technical indicators')
    dataset_building.add_argument('--btc_price',action='store_false', help='remove BTC price oracle')
    dataset_building.add_argument('--btc_indicators',action='store_false', help='remove BTC price technical indicators')
    dataset_building.add_argument('--usd_index',action='store_false', help='remove USD index oracle')
    dataset_building.add_argument('--usd_indicators',action='store_false', help='remove USD index technical indicators')
    dataset_building.add_argument('--swap_size',action='store_false', help='remove Uniswap swap size metrics')
    dataset_building.add_argument('--fear_greed',action='store_false', help='remove Fear and Greed index')
    dataset_building.add_argument('--gegen',action='store_false', help='remove Gegenbauer liquidity curve scores')
    dataset_building.add_argument('--gegen_indicators',action='store_false', help='remove Gegenbauer liquidity curve time series features')

    class_target = parser.add_argument_group('classification target arguments')
    class_target.add_argument('-t','--target', action='store_true', help='add binary classification target for depeg event within target window')
    class_target.add_argument('-w','--target_window', type=int, default=24, help='time window (in hours) for classification target')
    class_target.add_argument('-th','--target_threshold', type=int, default=25, help='threshold (in bps) for classification target')
    class_target.add_argument('-ds','--depeg_side', type=str, default='both', choices=['both', 'up', 'down'], help='depeg side for classification target')
    class_target.add_argument('-dt','--dynamic_threshold', action='store_true', help='use dynamic threshold for classification target')
    
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main file arguments')
    parser = add_dataset_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    dataset_path = build_dataset(**dict_args)



    