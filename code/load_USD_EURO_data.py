import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf

def fetch_data(ticker):
    yf.pdr_override()
    df = pdr.get_data_yahoo(ticker, start='2005-07-12', end='2017-11-03')
    close_data = df['Close'].values
    return close_data

def save_data_to_csv(data, filename):
    np.savetxt(filename, data, delimiter=',')

def load_data_from_csv(filename):
    return np.genfromtxt(filename, delimiter=',')

def main():
    usd_eur_ticker = 'EURUSD=X'
    data = fetch_data(usd_eur_ticker)
    save_data_to_csv(data, 'USD_EUR.csv')
    # loaded_data = load_data_from_csv('USD_EUR.csv')
    # print(loaded_data.shape)

if __name__ == "__main__":
    main()
