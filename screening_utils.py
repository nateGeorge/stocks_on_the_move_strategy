import sys

import talib
import numpy as np

sys.path.append('../stock_prediction/code')
import dl_quandl_EOD as dlq

sys.path.append('../beat_market_analysis/code')
import constituents_utils as cu



stocks = dlq.load_stocks()


# using SP600
# first check if index is bullish; if price is above 200 SMA
sly = stocks['SLY']
ijr = stocks['IJR']
vioo = stocks['VIOO']
sly['200d_sma'] = talib.SMA(sly['Adj_Close'].values, timeperiod=200)
ijr['200d_sma'] = talib.SMA(ijr['Adj_Close'].values, timeperiod=200)
vioo['200d_sma'] = talib.SMA(vioo['Adj_Close'].values, timeperiod=200)
# take majority vote
last_sly = sly.iloc[-1]
last_ijr = ijr.iloc[-1]
last_vioo = vioo.iloc[-1]
sly_above = last_sly['Adj_Close'] > last_sly['200d_sma']
ijr_above = last_ijr['Adj_Close'] > last_ijr['200d_sma']
vioo_above = last_vioo['Adj_Close'] > last_vioo['200d_sma']

is_bullish = False
if sum([sly_above, ijr_above, vioo_above]) > 1:
    is_bullish = True
    print('sp600 bullish')

# TODO: plot the indexes with their SMAs

# if bullish, we should buy stocks from ranked list
if is_bullish:
    # get current index constituents
    barchart_const = cu.load_sp600_files()
    tickers = barchart_const.index

    # get volatility-weighted exponential fit to data to rank stocks
    for t in tickers:
        df = stocks[t]
        # ln of last 90 days of closes
