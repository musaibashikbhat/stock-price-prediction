import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def preprocess_data(filename):
    """
    Load and preprocess stock data from a CSV file.
    
    Parameters:
    filename (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame
    """
    df = pd.read_csv(filename, index_col='Date', parse_dates=True)
    
    # Example preprocessing: Use only the 'Close' price for simplicity
    df = df[['Close']]
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    
    return df

def train_model(X_train, y_train):
    """
    Train a linear regression model.
    
    Parameters:
    X_train (pd.DataFrame): Training features
    y_train (pd.Series): Training target
    
    Returns:
    LinearRegression: Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print metrics.
    
    Parameters:
    model (LinearRegression): Trained model
    X_test (pd.DataFrame): Test features
    y_test (pd.Series): Test target
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    
    # Plot predictions vs true values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label='True Values', color='blue')
    plt.plot(y_test.index, predictions, label='Predictions', color='red')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Stock Price Prediction')
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    df = preprocess_data('aapl_stock_data.csv')
    
    # Prepare data for training
    X = df[['Close']]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
