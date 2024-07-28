import pandas as pd


if __name__ == '__main__':


    file_path = './processed_data/AAPL_daily_data_20240701_230513_processed.csv'

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Display the first few rows of the DataFrame to inspect
    print(df.head())

    # Print the DataFrame's information to check data types and non-null counts
    print(df.info())

    # Optionally, check the statistics of the DataFrame to understand distributions
    print(df.describe())
