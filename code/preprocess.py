import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf

def get_bank_list():
    return [
        "AXISBANK.BO", "BANKBARODA.BO", "FEDERALBNK.BO", "HDFCBANK.BO",
        "ICICIBANK.BO", "INDUSINDBK.BO", "KOTAKBANK.BO", "PNB.BO",
        "SBIN.BO", "YESBANK.BO"
    ]

def fetch_data(bank_list):
    yf.pdr_override()
    data_list = []

    for bank in bank_list:
        df = pdr.get_data_yahoo(bank, start='2005-07-12', end='2017-11-03')
        close_data = df['Close'].values

        if close_data.shape[0] > 3033:
            data_list.append(close_data[3:3035])
        else:
            data_list.append(close_data[0:3032])

    return np.array(data_list)

def save_data_to_csv(data, filename):
    np.savetxt(filename, data, delimiter=',')

def load_data_from_csv(filename):
    return np.genfromtxt(filename, delimiter=',')

def main():
    bank_list = get_bank_list()
    data = fetch_data(bank_list)
    save_data_to_csv(data, 'BANKEX.csv')
    loaded_data = load_data_from_csv('BANKEX.csv')
    print(loaded_data.shape)

if __name__ == "__main__":
    main()
