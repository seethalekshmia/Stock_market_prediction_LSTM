import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go

class StockPredictionApp:
    def __init__(self):
        st.set_page_config(page_title='Stock Price Prediction', page_icon=':chart_with_upwards_trend:')
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'scaler' not in st.session_state:
            st.session_state.scaler = None
    
    def load_and_preprocess_data(self, uploaded_file):
        """
        Load and preprocess uploaded stock market data
        """
        try:
            stock_data = pd.read_csv(uploaded_file)
            
            # Clean column names by stripping extra spaces and special characters
            stock_data.columns = stock_data.columns.str.strip()
            
            # Define possible column names and keywords
            column_mapping = {
                'Date': ['Date'],
                'Open': ['Open', 'Shares Traded', 'Opening'],
                'Close': ['Close', 'Closing', 'Last']
            }
            
            # Normalize column names by checking for expected keywords
            normalized_columns = {}
            for standard_col, possible_names in column_mapping.items():
                for possible_name in possible_names:
                    if possible_name in stock_data.columns:
                        normalized_columns[standard_col] = possible_name
                        break
            
            # Check if required columns were found
            missing_columns = [col for col in ['Date', 'Open', 'Close'] if col not in normalized_columns]
            if missing_columns:
                st.error(f"Missing columns in the uploaded file: {', '.join(missing_columns)}")
                return None
            
            # Rename columns to the standardized names
            stock_data = stock_data.rename(columns=normalized_columns)
            
            # Convert Date to datetime format
            stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce').dt.date
            stock_data = stock_data.dropna(subset=['Date'])
            stock_data.set_index('Date', drop=True, inplace=True)
            
            # Clean numeric columns using pd.to_numeric to handle errors
            stock_data['Open'] = pd.to_numeric(stock_data['Open'].replace({'\\$': '', ',': ''}, regex=True), errors='coerce')
            stock_data['Close'] = pd.to_numeric(stock_data['Close'].replace({'\\$': '', ',': ''}, regex=True), errors='coerce')
            
            # Drop rows with NaN values after conversion
            stock_data = stock_data.dropna()
            
            return stock_data
        except Exception as e:
            st.error(f"Error processing the file: {e}")
            return None
    
    def create_sequence(self, dataset, sequence_length=50):
        """
        Create sequences for prediction
        """
        sequences = []
        
        for stop_idx in range(sequence_length, len(dataset)):
            start_idx = stop_idx - sequence_length
            sequences.append(dataset.iloc[start_idx:stop_idx])
        
        return np.array(sequences)
    
    def predict_next_prices(self, stock_data):
        """
        Make predictions for next stock prices
        """
        try:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(stock_data)
            
            model = tf.keras.models.load_model('stock_prediction_model.h5')
            
            prediction_seq = self.create_sequence(pd.DataFrame(scaled_data, columns=stock_data.columns, index=stock_data.index))
            
            predicted_scaled = model.predict(prediction_seq[-1:])
            
            predicted_prices = scaler.inverse_transform(predicted_scaled)[0]
            
            return predicted_prices
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def plot_stock_prices(self, stock_data):
        """
        Create interactive plot of stock prices
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=stock_data.index, 
            y=stock_data['Open'], 
            mode='lines', 
            name='Open Price',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=stock_data.index, 
            y=stock_data['Close'], 
            mode='lines', 
            name='Close Price',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Stock Price History',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)
    
    def run(self):
        """
        Main Streamlit app
        """
        st.title('🚀 Stock Price Prediction App')
        
        st.sidebar.header('Upload Stock Data')
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file", 
            type=['csv'], 
            help="Upload a CSV file with 'Date', 'Open', and 'Close' columns"
        )
        
        if uploaded_file is not None:
            stock_data = self.load_and_preprocess_data(uploaded_file)
            
            if stock_data is not None:
                st.subheader('Data Overview')
                st.dataframe(stock_data.head())
                
                self.plot_stock_prices(stock_data)
                
                st.sidebar.subheader('Price Prediction')
                if st.sidebar.button('Predict Next Day Prices'):
                    predicted_prices = self.predict_next_prices(stock_data)
                    
                    if predicted_prices is not None:
                        st.sidebar.success(f"Predicted Open Price: ${predicted_prices[0]:.2f}")
                        st.sidebar.success(f"Predicted Close Price: ${predicted_prices[1]:.2f}")
                        
                        last_open = stock_data['Open'].iloc[-1]
                        last_close = stock_data['Close'].iloc[-1]
                        
                        open_change = predicted_prices[0] - last_open
                        close_change = predicted_prices[1] - last_close
                        
                        st.sidebar.info(f"Open Price Change: ${open_change:.2f}")
                        st.sidebar.info(f"Close Price Change: ${close_change:.2f}")

# Run the Streamlit app
def main():
    app = StockPredictionApp()
    app.run()

if __name__ == '__main__':
    main()
