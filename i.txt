import yfinance as yf
import pandas as pd

def download_stock_data(ticker, start_date, end_date, filename):
    """
    Download historical stock data and save it to a CSV file.
    
    Parameters:
    ticker (str): Stock symbol (e.g., 'AAPL' for Apple)
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    filename (str): Path to save the CSV file
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}")
    
    # Download the stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Save to CSV
    stock_data.to_csv(filename)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Example usage
    download_stock_data('AAPL', '2023-01-01', '2024-01-01', 'aapl_stock_data.csv')
