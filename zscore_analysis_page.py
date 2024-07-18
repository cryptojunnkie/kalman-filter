import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from pykalman import KalmanFilter
from datetime import datetime, timedelta

def page_zscore_analysis():
    # Set the title of the page
    st.title('Stock Z-Score Analysis Signals')

    # Function to generate signals with specific names for Buy and Take Profit levels
    def generate_signals(z_scores):
        signals = []
        for i in range(1, len(z_scores)):
            prev_z_score = z_scores[i - 1]
            curr_z_score = z_scores[i]
            if prev_z_score <= -4 and curr_z_score > -4:  # Buy 3
                signals.append('Buy Signal (Buy 4)')
            elif prev_z_score <= -3 and curr_z_score > -3:  # Buy 3
                signals.append('Buy Signal (Buy 3)')
            elif prev_z_score <= -1 and curr_z_score > -1:  # Buy 2
                signals.append('Buy Signal (Buy 2)')
            elif prev_z_score <= -0.4 and curr_z_score > -0.4:  # Buy 1
                signals.append('Buy Signal (Buy 1)')
            elif prev_z_score >= 0.25 and curr_z_score < 0.25:  # TP 1
                signals.append('Take Profit Signal (TP 1)')
            elif prev_z_score >= 0.5 and curr_z_score < 0.5:  # TP 2
                signals.append('Take Profit Signal (TP 2)')
            elif prev_z_score >= 1 and curr_z_score < 1:  # TP 3
                signals.append('Take Profit Signal (TP 3)')
            else:
                signals.append('No Signal')
        return signals

    # User input for multiple stock tickers
    tickers_input = st.text_input('Enter stock tickers (separated by commas)', 'AAPL, GOOGL, CDE, DJT, AVGO, OLPX, SIRI, RUM, XLK, BTG, NKLA, TSLA, RIOT, PACB')

    # Split input tickers and remove any spaces
    tickers = [ticker.strip() for ticker in tickers_input.split(',')]

    # Initialize empty lists for potential take profit and potential buy entry
    potential_take_profit = []
    potential_buy_entry = []

    # Initialize an empty DataFrame to store values
    data_rows = []
    for ticker in tickers:
        # Retrieve historical stock price data for each ticker and get the long name
        stock_data = yf.download(ticker, start=datetime.today()-timedelta(days=1000), end=datetime.today())
        long_name = yf.Ticker(ticker).info.get('longName')

        close_prices = stock_data["Close"].values

        # Initialize Kalman Filter
        kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1],
                        initial_state_mean=0, initial_state_covariance=1,
                        observation_covariance=1, transition_covariance=0.01)

        # Apply Kalman Filter to estimate moving average
        state_means, _ = kf.filter(close_prices)
        kalman_avg = state_means.flatten()

        # Calculate Z-Score
        std_dev = np.std(close_prices)
        z_score = (close_prices - kalman_avg) / std_dev

        # Generate signals based on Z-score conditions
        signals = generate_signals(z_score)

        # Add values to the DataFrame
        data_rows.append([long_name, ticker, close_prices[-1], kalman_avg[-1], z_score[-1], signals[-1],
                        z_score[-2] if len(z_score) > 1 else None])

        # Display stock price, Kalman estimated moving average, and company's long name
        #st.subheader(f'{long_name} ({ticker})')
        #st.line_chart(stock_data["Close"])
        close_prices = stock_data["Close"].values

        # Calculate moving average on the close prices
        moving_avg = pd.Series(close_prices).rolling(window=20).mean()

        # Create a Plotly figure for the price chart with moving average
        price_chart = go.Figure()
        price_chart.add_trace(go.Scatter(x=stock_data.index, y=close_prices, mode='lines', name='Close Price', line=dict(width=1)))
        price_chart.add_trace(go.Scatter(x=stock_data.index, y=moving_avg, mode='lines', name='Moving Average', line=dict(color='red')))

        # Update layout of the price chart
        price_chart.update_layout(
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis=dict(rangeslider=dict(visible=True))
    )

        # Display the price chart with moving average
        st.write('### Stock Price with Moving Average')
        st.subheader(f'{long_name} ({ticker})')
        st.plotly_chart(price_chart)

        # Create an interactive Z-Score chart using Plotly for each ticker
        z_score_chart = go.Figure()
        z_score_chart.add_trace(go.Scatter(x=stock_data.index, y=z_score, mode='lines', name='Z-Score', line=dict(width=1),
                        hovertemplate="Date: %{x}<br>Z-Score: %{y:.2f}<br><b>Market Price</b>: %{customdata[0]:.2f}"))
        
        for i, signal in enumerate(signals):
            d = stock_data.index[i].date()
            next_day = d + timedelta(days=1)

            if 'Buy' in signal:
                z_score_chart.add_trace(go.Scatter(x=[next_day], y=[z_score[i]], mode='markers', name=f'{signal} ({d.strftime("%Y-%m-%d")})', 
                                    marker=dict(color='green', size=7), 
                                    hovertemplate=f"Date: {next_day}<br>Z-Score: {z_score[i]:.2f}<br><b>Market Price</b>: {close_prices[i]:.2f}"))
            elif 'Take Profit' in signal:
                z_score_chart.add_trace(go.Scatter(x=[next_day], y=[z_score[i]], mode='markers', name=f'{signal} ({d.strftime("%Y-%m-%d")})', 
                                    marker=dict(color='red', size=7), 
                                    hovertemplate=f"Date: {next_day}<br>Z-Score: {z_score[i]:.2f}<br><b>Market Price</b>: {close_prices[i]:.2f}"))
            
        # Update layout to include legend with fixed item size for a scrollable effect
        z_score_chart.update_layout(legend=dict(itemwidth=80, itemsizing='constant'))


        # Add individual green dotted lines at -0.5, -1, -3, -4
        for level in [-0.4, -1, -3, -4]:
            z_score_chart.add_shape(
                type="line",
                x0=min(stock_data.index),
                x1=max(stock_data.index),
                y0=level,
                y1=level,
                line=dict(
                    color="green",
                    width=1,
                    dash="dot"
                )
            )

        # Add individual red dotted lines at 0.25, 0.5, and 1
        for level in [0.25, 0.5, 1]:
            z_score_chart.add_shape(
                type="line",
                x0=min(stock_data.index),
                x1=max(stock_data.index),
                y0=level,
                y1=level,
                line=dict(
                    color="red",
                    width=1,
                    dash="dot"
                )
            )

        z_score_chart.update_layout(title=f'Z-Score with Entry (Buy) and Take Profit Zones for {ticker}', xaxis_title='Time', yaxis_title='Z-Score')
        # Add x-axis slider for chart expansion
        z_score_chart.update_layout(xaxis=dict(rangeslider=dict(visible=True)))

        st.write('### Signals')
        st.write(signals[-1])

        st.plotly_chart(z_score_chart)

    # Increase the maximum column width for better display
    pd.set_option('display.max_colwidth', 2000)

    # Create a DataFrame with the collected values
    df = pd.DataFrame(data_rows, columns=['Company Name', 'Ticker', 'Closing Price', 'Kalman Avg', 'Previous Z-Score', 'Signal', 'Z-Score'])

    # Display the DataFrame as a summary table
    st.write('### Summary Table')
    st.dataframe(df)