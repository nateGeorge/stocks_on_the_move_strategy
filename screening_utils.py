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


def check_index_bullish(stocks, date='latest'):
    """
    stocks is a dictionary with tickers as keys and dataframes as values,
    from dl_quandl_EOD

    date should be either 'latest', or a datestring, like '01-31-2018'
    """
    sly = stocks['SLY']
    ijr = stocks['IJR']
    vioo = stocks['VIOO']
    sly['200d_sma'] = talib.SMA(sly['Adj_Close'].values, timeperiod=200)
    ijr['200d_sma'] = talib.SMA(ijr['Adj_Close'].values, timeperiod=200)
    vioo['200d_sma'] = talib.SMA(vioo['Adj_Close'].values, timeperiod=200)
    if date == 'latest':
        # take majority vote
        last_sly = sly.iloc[-1]
        last_ijr = ijr.iloc[-1]
        last_vioo = vioo.iloc[-1]
    else:
        last_sly = sly[date]
        last_ijr = ijr[date]
        last_vioo = vioo[date]

    sly_above = last_sly['Adj_Close'] > last_sly['200d_sma']
    ijr_above = last_ijr['Adj_Close'] > last_ijr['200d_sma']
    vioo_above = last_vioo['Adj_Close'] > last_vioo['200d_sma']

    is_bullish = False
    if sum([sly_above, ijr_above, vioo_above]) > 1:
        is_bullish = True
        print('sp600 bullish')
    else:
        print('sp600 not bullish')

    return is_bullish


def get_current_buylist(acct_val=20000, risk_factor=0.0012):
    stocks = dlq.load_stocks()

    # using SP600
    # first check if index is bullish; if price is above 200 SMA
    is_bullish = check_index_bullish(stocks)

    # TODO: plot the indexes with their SMAs
    rank_df = pd.DataFrame()

    # if bullish, we should buy stocks from ranked list
    if is_bullish:
        # get current index constituents
        barchart_const = cu.load_sp600_files()
        tickers = [t.replace('.', '_') for t in barchart_const.index]  # quandl data has underscores instead of periods

        # get volatility-weighted exponential fit to data to rank stocks
        for t in tqdm(tickers):
            one_df = calc_latest_metrics(stocks[t], t)
            rank_df = rank_df.append(one_df)

    filtered_df = rank_df[(rank_df['bullish'] == True) & (rank_df['gap'] == False)].sort_values(by='rank_score', ascending=False)
    filtered_df = get_cost_shares_etc(filtered_df, acct_val=acct_val, risk_factor=risk_factor)

    to_buy = filtered_df[filtered_df['cumulative_cost'] <= (acct_val - 100)]  # save $100 for commissions
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

    return to_buy


def get_cost_shares_etc(df, acct_val=20000, risk_factor=0.0012):
    """
    calculates shares, cost, weights

    risk factor suggested from book to be in range of 8-15 basis points, or 0.0008 - 0.0015
    adjust so have 20-30 stocks suggested, no more than 50 (or 10% of index) or might start approximating index too well

    for sp600, 0.001 seemed to give about 35 stocks, not too bad
    """
    # TODO: get account value from IB
    # Shares = account_value * risk_factor / ATR
    df['shares'] = acct_val * risk_factor / df['atr']
    # round down to be conservative
    df['rounded_shares'] = df['shares'].apply(lambda x: int(x))
    df['cost'] = df['rounded_shares'] * df['Adj_Close']
    df['cumulative_cost'] = df['cost'].cumsum()
    df['weight'] = df['Adj_Close'] * df['rounded_shares'] / acct_val
    df['cumulative_weight'] = df['weight'].cumsum()
    return df


def calc_latest_metrics(df, ticker):
    """
    df is the stock dataframe
    ticker is the ticker symbol (string)

    """
    # calculate some metrics/signals first
    # stocks[ticker]['100day_SMA'] = talib.SMA(stocks[ticker]['Adj_Close'].values, timeperiod=100)
    sma_100 = talib.SMA(df.iloc[-300:]['Adj_Close'].values, timeperiod=100)
    # should be above 100 day SMA to buy
    bullish = True
    # if stocks[ticker].iloc[-1]['100day_SMA'] >= stocks[ticker].iloc[-1]['Adj_Close']:
    if sma_100[-1] >= df.iloc[-1]['Adj_Close']:
        # print('stock not above 100day SMA')
        bullish = False

    # check to make sure no gaps greater than 15% in last 90 days
    # todays open minus yesterdays close
    gaps = (df['Adj_Open'] - df['Adj_Close'].shift(1)) / df['Adj_Close'].shift(1)
    gap = False
    if any(abs(gaps[-90:]) > 0.15):
        # print('recent gap greater than 15%')
        gap = True

    # get 20-day ATR
    # stocks[ticker]['20d_ATR'] = talib.ATR(stocks[ticker]['Adj_High'].values, stocks[ticker]['Adj_Low'].values, stocks[ticker]['Adj_Close'].values, timeperiod=20)
    # because ATR depends on previous values, best to use all possible values
    atr_20d = talib.ATR(df['Adj_High'].values, df['Adj_Low'].values, df['Adj_Close'].values, timeperiod=20)
    atr = atr_20d[-1]

    # ln of last 90 days of closes
    # TODO: examine and deal with large jumps in price
    # maybe if large jump, only take price after that, or ignore stocks with large one-off jumps
    # which would be jumps with a few moves far outside the ATR
    ln_prices = np.log(df['Adj_Close'].iloc[-90:].values)
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
                            'Adj_Close': df.iloc[-1]['Adj_Close']},
                            index=[ticker])

    return one_df


def calc_all_metrics(stocks, ticker):
    """
    stocks is dictionary with tickers as keys and dfs as values
    ticker is a string with ticker symbol, like 'TSLA'

    this calculates the metrics for the ticker and adds them to the dataframe,
    useful for backtesting
    """
    pass



def portfolio_rebalance(position_check=True, acct_val=20000, risk_factor=0.0012):
    """
    to be done once a week

    If stock is below 100 day MA, if had a 15% or more gap, left index, if no longer in 20% of rankings, sell it

    also does position resizing if position_check is True
    """
    stocks = dlq.load_stocks()

    # get current holdings from IB or stored list
    # use latest stored csv for now
    holdings_files = glob.glob('current_holdings_*.csv')
    daily_dates = [pd.to_datetime(f.split('/')[-1].split('_')[-1].split('.')[0]) for f in holdings_files]
    last_daily = np.argmax(daily_dates)
    latest_file = holdings_files[last_daily]
    df = pd.read_csv(latest_file, index_col=0)

    barchart_const = cu.load_sp600_files()
    tickers = set([t.replace('.', '_') for t in barchart_const.index])  # quandl data has underscores instead of periods

    # get index constituents
    full_df = pd.DataFrame()
    for t in df.index:
        one_df = calc_latest_metrics(stocks[t], t)
        full_df = full_df.append(one_df)

    full_df = full_df.sort_values(by='rank_score', ascending=False)
    top_20_pct = set(full_df.index[:120])

    kickout = full_df[(full_df['bullish'] == False) | (full_df['gap'] == True) | ([f not in top_20_pct.union(tickers) for f in full_df.index.tolist()])]
    if kickout.shape[0] > 0:
        print('liquidate:')
        print(kickout)
        # calculate money available for new purchases

        # get new purchases and save current_holdings file



    if position_check:
        # to do:
        full_df = get_cost_shares_etc(full_df)
        full_df['current_shares'] = df.loc[full_df.index]['rounded_shares']
        full_df['pct_diff_shares'] = (full_df['rounded_shares'] - full_df['current_shares']) / full_df['current_shares']
        to_rebalance = full_df[full_df['pct_diff_shares'].abs() >= 0.1]  # book suggested 5% as threshold for resizing, use 10% for less transaction cost
        if to_rebalance.shape[0] > 0:
            print('rebalance:')
            print(to_rebalance)


def save_one_day_df(stocks, date='latest', write_holdings_file=False, acct_val=20000, risk_factor=0.0012, reserve_for_commisions=100):
    """
    saves current holdings file for specified date

    stocks is a dictionary of stocks dataframes from EOD quandl data

    date should be a string with 'Y-m-d' like  '2018-09-24'
    """
    if date == 'latest':
        to_buy = get_current_buylist()
        return to_buy

    # dictionaries with Y-m-d date format as keys
    # gets tickers from WRDS
    # constituent_companies, constituent_tickers, unique_dates = cu.get_historical_constituents_wrds()
    # tickers = constituent_tickers[date].values
    # tickers = [t.replace('.', '_') for t in tickers]  # quandl data has underscores instead of periods
    barchart_const = cu.load_sp600_files(date=date)
    tickers = [t.replace('.', '_') for t in barchart_const.index]  # quandl data has underscores instead of periods

    rank_df = pd.DataFrame()
    # get volatility-weighted exponential fit to data to rank stocks
    for t in tqdm(tickers):
        if stocks[t].shape[0] < 100:
            print('stock too new')
            continue

        one_df = calc_latest_metrics(stocks[t].loc[:date], t)
        rank_df = rank_df.append(one_df)

    filtered_df = rank_df[(rank_df['bullish'] == True) & (rank_df['gap'] == False)].sort_values(by='rank_score', ascending=False)
    filtered_df = get_cost_shares_etc(filtered_df, acct_val=acct_val, risk_factor=risk_factor)

    to_buy = filtered_df[filtered_df['cumulative_cost'] <= (acct_val - reserve_for_commisions)]  # save $100 for commisions
    money_left = acct_val - to_buy['cost'].sum()
    next_stock = filtered_df[filtered_df['cumulative_cost'] > acct_val].iloc[0]
    next_stock['rounded_shares'] = money_left // next_stock['Adj_Close']
    next_stock['cost'] = next_stock['rounded_shares'] * next_stock['Adj_Close']
    to_buy = to_buy.append(next_stock.to_frame().T)
    to_buy['cumulative_cost'] = to_buy['cost'].cumsum()

    # save for later reference
    # today_ny = datetime.datetime.now(pytz.timezone('America/New_York')).strftime('%m-%d-%Y')
    to_buy.to_csv('to_buy_' + date + '.csv')
    if write_holdings_file:
        # writes file as current holdings
        to_buy.to_csv('current_holdings_' + date + '.csv')

    rank_df.to_csv('rank_df_' + date + '.csv')
    filtered_df.to_csv('filtered_df_' + date + '.csv')
    print(to_buy)

    return to_buy


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


# TODO: get current holdings from IB and do portfolio rebalancing and position resizing

# date = '2018-09-21'
# stocks = dlq.load_stocks()
# save_one_day_df(stocks, date)
# to_buy = save_one_day_df(stocks, date='2018-09-21', write_holdings_file=True, risk_factor=0.001, reserve_for_commisions=0)
