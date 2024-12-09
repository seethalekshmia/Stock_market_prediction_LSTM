import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_preprocess_data(file_path):
    """
    Load and preprocess stock market data
    """
    gstock_data = pd.read_csv(file_path)
    gstock_data = gstock_data.rename(columns={'Close/Last': 'Close'})
    gstock_data = gstock_data[['Date', 'Open', 'Close']]
    
    gstock_data['Date'] = pd.to_datetime(gstock_data['Date']).dt.date
    gstock_data.set_index('Date', drop=True, inplace=True)
    
    gstock_data['Open'] = gstock_data['Open'].replace({'\\$': '', ',': ''}, regex=True).astype(float)
    gstock_data['Close'] = gstock_data['Close'].replace({'\\$': '', ',': ''}, regex=True).astype(float)
    
    # Return both data and scaler for future use
    scaler = MinMaxScaler()
    gstock_data[['Open', 'Close']] = scaler.fit_transform(gstock_data[['Open', 'Close']])
    
    return gstock_data, scaler

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

def create_sequence(dataset, sequence_length=60):
    """
    Create sequences with enhanced feature engineering
    """
    # Create a copy of the dataset to avoid warnings
    df = dataset.copy()
    
    # Technical indicators
    df.loc[:, 'Returns'] = df['Close'].pct_change()
    df.loc[:, 'MA5'] = df['Close'].rolling(window=5).mean()
    df.loc[:, 'MA20'] = df['Close'].rolling(window=20).mean()
    df.loc[:, 'Volatility'] = df['Returns'].rolling(window=20).std()
    df.loc[:, 'RSI'] = calculate_rsi(df['Close'])
    df.loc[:, 'Price_Range'] = (df['Close'] - df['Open']).abs()
    
    # Remove NaN values
    df = df.dropna()
    
    # Normalize all features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    sequences = []
    labels = []
    
    for i in range(len(df_scaled) - sequence_length):
        sequences.append(df_scaled.iloc[i:(i + sequence_length)].values)
        labels.append(df_scaled.iloc[i + sequence_length][['Open', 'Close']].values)
    
    return np.array(sequences), np.array(labels)

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def build_model(input_shape):
    """
    Build an enhanced LSTM model with attention mechanism
    """
    model = Sequential()
    
    # First Bidirectional LSTM layer with more units
    model.add(Bidirectional(LSTM(units=256,
                                return_sequences=True,
                                input_shape=input_shape)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # Second Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(units=128,
                                return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # Third Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(units=64)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # Dense layers for final prediction
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(2))

    # Use Adam optimizer with custom learning rate and clipnorm
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    )
    
    model.compile(loss='huber',  # Huber loss is more robust to outliers
                 optimizer=optimizer,
                 metrics=['mean_absolute_error'])
    
    return model

def main():
    # File path
    file_path = "/home/hp/Documents/deep_learning/deep_learning_project/HistoricalData_1731575648658.csv"
    
    # Load and preprocess data
    gstock_data, scaler = load_and_preprocess_data(file_path)
    
    # Split into training and testing sets (using 85% for training)
    training_size = int(len(gstock_data) * 0.85)
    train_data = gstock_data[:training_size]
    test_data = gstock_data[training_size:]
    
    # Create sequences with improved features
    train_seq, train_label = create_sequence(train_data)
    test_seq, test_label = create_sequence(test_data)
    
    # Build model
    model = build_model((train_seq.shape[1], train_seq.shape[2]))
    
    # Early stopping callback with longer patience
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    # Learning rate scheduler with slower decay
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=7,
        min_lr=0.00001
    )
    
    # Train the model with improved parameters
    history = model.fit(
        train_seq, 
        train_label,
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Make predictions
    test_predicted = model.predict(test_seq)
    
    # Inverse scale predictions
    test_inverse_actual = scaler.inverse_transform(test_label)
    test_inverse_predicted = scaler.inverse_transform(test_predicted)
    
    # Save the model
    model.save('stock_prediction_model.h5')
    
    # Calculate and print metrics
    mae = mean_absolute_error(test_inverse_actual, test_inverse_predicted)
    rmse = np.sqrt(mean_squared_error(test_inverse_actual, test_inverse_predicted))
    r2 = r2_score(test_inverse_actual, test_inverse_predicted)
    
    mse_open = mean_squared_error(test_inverse_actual[:, 0], test_inverse_predicted[:, 0])
    mse_close = mean_squared_error(test_inverse_actual[:, 1], test_inverse_predicted[:, 1])
    
    print("\nEvaluation Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE (Open): {mse_open:.4f}")
    print(f"MSE (Close): {mse_close:.4f}")
    print(f"R²: {r2:.4f}")

if __name__ == "__main__":
    main()
