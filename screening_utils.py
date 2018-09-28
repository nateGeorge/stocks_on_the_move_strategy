import sys
import datetime
import glob

import talib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

sys.path.append('../stock_prediction/code')
import dl_quandl_EOD as dlq

sys.path.append('../beat_market_analysis/code')
import constituents_utils as cu


def get_current_buylist():
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
    rank_df = pd.DataFrame()

    # if bullish, we should buy stocks from ranked list
    if is_bullish:
        # get current index constituents
        barchart_const = cu.load_sp600_files()
        tickers = [t.replace('.', '_') for t in barchart_const.index]  # quandl data has underscores instead of periods

        # get volatility-weighted exponential fit to data to rank stocks
        for t in tqdm(tickers):
            # calculate some metrics/signals first
            stocks[t]['100day_SMA'] = talib.SMA(stocks[t]['Adj_Close'].values, timeperiod=100)
            # should be above 100 day SMA to buy
            bullish = True
            if stocks[t].iloc[-1]['100day_SMA'] >= stocks[t].iloc[-1]['Adj_Close']:
                # print('stock not above 100day SMA')
                bullish = False

            # check to make sure no gaps greater than 15% in last 90 days
            # todays open minus yesterdays close
            stocks[t]['gaps'] = (stocks[t]['Adj_Open'] - stocks[t]['Adj_Close'].shift(1)) / stocks[t]['Adj_Close'].shift(1)
            gap = False
            if any(abs(stocks[t]['gaps'].iloc[-90:]) > 0.15):
                # print('recent gap greater than 15%')
                gap = True

            # get 20-day ATR
            stocks[t]['20d_ATR'] = talib.ATR(stocks[t]['Adj_High'].values, stocks[t]['Adj_Low'].values, stocks[t]['Adj_Close'].values, timeperiod=20)
            atr = stocks[t]['20d_ATR'].iloc[-1]

            # ln of last 90 days of closes
            ln_prices = np.log(stocks[t]['Adj_Close'].iloc[-90:].values)
            lr = LinearRegression()
            X = np.arange(ln_prices.shape[0]).reshape(-1, 1)
            lr.fit(X, ln_prices)
            # TODO: check that this shouldn't be np.exp(lr.coef_[0])
            slope = lr.coef_[0]  # should be approximately how much in pct price changes per day
            annualized_slope = (1 + slope) ** 250
            r2 = lr.score(X, ln_prices)
            rank_score = annualized_slope * r2

            # TODO: plot fits

            one_df = pd.DataFrame({'bullish': bullish,
                                    'gap': gap,
                                    'atr': atr,
                                    'slope': annualized_slope,
                                    'r2': r2,
                                    'rank_score': rank_score,
                                    'Adj_Close': stocks[t].iloc[-1]['Adj_Close']},
                                    index=[t])
            rank_df = rank_df.append(one_df)

    filtered_df = rank_df[(rank_df['bullish'] == True) & (rank_df['gap'] == False)].sort_values(by='rank_score', ascending=False)
    # Shares = account_value * risk_factor / ATR
    acct_val = 20000  # TODO: get account value from IB
    risk_factor = 0.001  # suggested from book to be in range of 8-15 basis points, or 0.0008 - 0.0015
    filtered_df['shares'] = acct_val * risk_factor / filtered_df['atr']
    # round down to be conservative
    filtered_df['rounded_shares'] = filtered_df['shares'].apply(lambda x: int(x))
    filtered_df['cost'] = filtered_df['rounded_shares'] * filtered_df['Adj_Close']
    filtered_df['cumulative_cost'] = filtered_df['cost'].cumsum()
    filtered_df['weight'] = filtered_df['Adj_Close'] * filtered_df['rounded_shares'] / acct_val
    filtered_df['cumulative_weight'] = filtered_df['weight'].cumsum()

    to_buy = filtered_df[filtered_df['cumulative_cost'] <= acct_val]
    money_left = acct_val - to_buy['cost'].sum()
    next_stock = filtered_df[filtered_df['cumulative_cost'] > acct_val].iloc[0]
    next_stock['rounded_shares'] = money_left // next_stock['Adj_Close']
    next_stock['cost'] = next_stock['rounded_shares'] * next_stock['Adj_Close']
    to_buy = to_buy.append(next_stock.to_frame().T)
    to_buy['cumulative_cost'] = to_buy['cost'].cumsum()

    # save for later reference
    # today_ny = datetime.datetime.now(pytz.timezone('America/New_York')).strftime('%m-%d-%Y')
    last_date = stocks[t].index[-1].strftime('%m-%d-%Y')
    to_buy.to_csv('to_buy_' + last_date + '.csv')
    rank_df.to_csv('rank_df_' + last_date + '.csv')
    filtered_df.to_csv('filtered_df_' + last_date + '.csv')
    print(to_buy)



def portfolio_rebalance():
    """
    to be done once a week

    If stock is below 100 day MA, if had a 15% or more gap, left index, if no longer in 20% of rankings, sell it
    """
    stocks_df = dlq.load_stocks()

    # get current holdings from IB or stored list
    holdings_files = glob.iglob('current_holdings*.csv')
    daily_dates = [pd.to_datetime(f.split('/')[-1].split('_')[-1].split('.')[0]) for f in daily_files]
    last_daily = np.argmax(daily_dates)
    latest_file = holdings_files[last_daily]
    df = pd.read_csv(latest_file)
    for t in df.index:
        pass
        # check if price below 100 day MA
        # TODO: make function to get TAs and check things, then return small df

    # TODO: get current holdings from IB and do portfolio rebalancing and position resizing
    # check if still bullish (above 100 day MA)
    # check for recent gaps (any above 15% in last 90 days)
    # check if still in top 20% of index
    # check if still in index


def save_first_day():
    # manually set current portfolio from first day
    datadict = {}
    datadict['tickers'] = ['AAON',
                            'AMED',
                            'ATNI',
                            'AVAV',
                            'BEAT',
                            'BELFB',
                            'CBM',
                            'COKE',
                            'CORE',
                            'CTRL',
                            'EHTH',
                            'ENDP',
                            'EPAY',
                            'FOXF',
                            'HF',
                            'HLIT',
                            'HMSY',
                            'INGN',
                            'IRDM',
                            'JBT',
                            'LGND',
                            'LXU',
                            'MRCY',
                            'OMCL',
                            'REGI',
                            'SPSC',
                            'SRDX',
                            'THRM',
                            'TREX',
                            'TRHC',
                            'UEIC',
                            'USPH',
                            'VSI']
    datadict['rounded_shares'] = [15, 6, 11, 4, 9, 16, 11, 4, 17, 16, 16, 31, 11,
                                10, 18, 105, 27, 2, 27, 6, 2, 58, 12, 11, 20, 8,
                                8, 16, 7, 5, 14, 5, 27]
    # get prices on last Friday
    last_prices = []
    for t in datadict['tickers']:
        last_prices.append(stocks_df[t].iloc[-1]['Adj_Close'])

    acct_val = 20000
    datadict['Adj_Close'] = last_prices
    df = pd.DataFrame(datadict)
    df.set_index('tickers', inplace=True)
    df['cost'] = df['Adj_Close'] * df['rounded_shares']
    df['weight'] = df['cost'] / acct_val
    today = today_ny = datetime.datetime.now(pytz.timezone('America/New_York')).strftime('%m-%d-%Y')
    df.to_csv('current_holdings_' + today + '.csv')
