import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_preprocess_data(file_path):
    """
    Load and preprocess stock market data
    """
    # Load data
    gstock_data = pd.read_csv(file_path)
    
    # Rename Close/Last to Close
    gstock_data = gstock_data.rename(columns={'Close/Last': 'Close'})
    
    # Select specific columns
    gstock_data = gstock_data[['Date', 'Open', 'Close']]
    
    # Convert Date to datetime and set as index
    gstock_data['Date'] = pd.to_datetime(gstock_data['Date']).dt.date
    gstock_data.set_index('Date', drop=True, inplace=True)
    
    # Remove dollar signs and convert to float
    gstock_data['Open'] = gstock_data['Open'].replace({'\\$': '', ',': ''}, regex=True).astype(float)
    gstock_data['Close'] = gstock_data['Close'].replace({'\\$': '', ',': ''}, regex=True).astype(float)
    
    return gstock_data

def plot_stock_prices(gstock_data):
    """
    Plot Open and Close prices
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    
    # First subplot for open prices
    ax[0].plot(gstock_data['Open'], label='Open', color='green')
    ax[0].set_xlabel('Date', size=15)
    ax[0].set_ylabel('Price($)', size=15)
    min_open = gstock_data['Open'].min()
    max_open = gstock_data['Open'].max()
    ax[0].set_yticks([min_open, max_open])
    ax[0].set_yticklabels([f'{min_open:.2f}', f'{max_open:.2f}'])
    ax[0].legend()
    
    # Second subplot for close prices
    ax[1].plot(gstock_data['Close'], label='Close', color='red')
    ax[1].set_xlabel('Date', size=15)
    ax[1].set_ylabel('Price($)', size=15)
    min_close = gstock_data['Close'].min()
    max_close = gstock_data['Close'].max()
    ax[1].set_yticks([min_close, max_close])
    ax[1].set_yticklabels([f'{min_close:.2f}', f'{max_close:.2f}'])
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig('stock_prices_comparison.png')
    plt.close()

def create_sequence(dataset, sequence_length=50):
    """
    Create sequences and labels from time series data.
    
    Parameters:
    -----------
    dataset : pandas.DataFrame
        Input time series dataset
    sequence_length : int, optional (default=50)
        Length of input sequences
    
    Returns:
    --------
    tuple: (sequences, labels)
        sequences: numpy array of input sequences
        labels: numpy array of corresponding labels
    """
    sequences = []
    labels = []
    
    # Create sliding window sequences
    for stop_idx in range(sequence_length, len(dataset)):
        start_idx = stop_idx - sequence_length
        
        # Sequence is the window of data before the label
        sequences.append(dataset.iloc[start_idx:stop_idx])
        
        # Label is the next data point after the sequence
        labels.append(dataset.iloc[stop_idx])
    
    return (np.array(sequences), np.array(labels))

def build_model(input_shape):
    """
    Build and compile the LSTM model
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50))
    model.add(Dense(2))
    
    model.compile(loss='mean_squared_error', 
                 optimizer='adam', 
                 metrics=['mean_absolute_error'])
    
    return model

def main():
    # File path
    file_path = "/home/hp/Documents/deep_learning/deep_learning_project/HistoricalData_1731575648658.csv"
    
    # Load and preprocess data
    gstock_data = load_and_preprocess_data(file_path)
    
    # Plot stock prices
    plot_stock_prices(gstock_data)
    
    # Scale the data
    global scaler
    scaler = MinMaxScaler()
    gstock_data[gstock_data.columns] = scaler.fit_transform(gstock_data)
    
    # Split into training and testing sets
    training_size = round(len(gstock_data) * 0.80)
    train_data = gstock_data[:training_size]
    test_data = gstock_data[training_size:]
    
    # Create sequences
    train_seq, train_label = create_sequence(train_data)
    test_seq, test_label = create_sequence(test_data)
    
    # Build and train model
    model = build_model((train_seq.shape[1], train_seq.shape[2]))
    model.summary()
    
    # Train the model
    history = model.fit(
        train_seq, 
        train_label, 
        epochs=80,
        validation_data=(test_seq, test_label), 
        verbose=1
    )
    
    # Make predictions
    test_predicted = model.predict(test_seq)
    
    # Inverse scale the actual values (test labels)
    test_inverse_actual = scaler.inverse_transform(test_label)
    
    # Inverse scale the predicted values
    test_inverse_predicted = scaler.inverse_transform(test_predicted)
    
    # Save the model
    model.save('stock_prediction_model.h5')
    
    print("\nModel has been trained and saved as 'stock_prediction_model.h5'")
    print("Predictions have been made on the test set")
    
    # Calculate MAE, RMSE, and R²
    mae = mean_absolute_error(test_inverse_actual, test_inverse_predicted)
    rmse = np.sqrt(mean_squared_error(test_inverse_actual, test_inverse_predicted))
    
    # Calculate baseline for both Open and Close
    baseline_open = np.mean(test_inverse_actual[:, 0])
    baseline_close = np.mean(test_inverse_actual[:, 1])

    # Calculate MAE for both Open and Close based on the baseline
    mae_baseline_open = mean_absolute_error(test_inverse_actual[:, 0], [baseline_open] * len(test_inverse_actual))
    mae_baseline_close = mean_absolute_error(test_inverse_actual[:, 1], [baseline_close] * len(test_inverse_actual))

    # Calculate MAS (Mean Absolute Scaled Error) for both Open and Close
    mas_open = mae / mae_baseline_open
    mas_close = mae / mae_baseline_close

    r2 = r2_score(test_inverse_actual, test_inverse_predicted)

    # Print results
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"MSE (Open): {mas_open}")
    print(f"MSE (Close): {mas_close}")
    print(f"R²: {r2}")

if __name__ == "__main__":
    main()
