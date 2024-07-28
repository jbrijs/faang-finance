
import pandas as pd

appl_splits = {
    '2020-08-28': 4,
    '2014-06-09': 7,
    '2005-02-28': 2,
    '2000-06-21': 2
}

def apply_splits(file_name, splits):
    df = pd.read_csv(file_name)
    

    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    
    for split_date, ratio in sorted(splits.items(), key=lambda x: pd.to_datetime(x[0])):
        split_date = pd.to_datetime(split_date)
        df.loc[df['time_stamp'] <= split_date, ['open', 'high', 'low', 'close']] /= ratio
        
    df['open'] = df['open'].round(2)
    df['high'] = df['high'].round(2)
    df['low'] = df['low'].round(2)
    df['close'] = df['close'].round(2)

    return df

def main():
    adjusted_df = apply_splits('./data/AAPL_daily_data.csv', appl_splits)
    adjusted_df.to_csv('./data/AAPL_daily_data_splits.csv', index=False)

if __name__ == '__main__':
    main()