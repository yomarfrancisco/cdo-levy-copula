import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import os
import ta  # Technical Analysis library
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

def create_features(ticker, start_date=None, end_date=None):
    """Create feature set for prediction"""
    if start_date is None:
        end_date = pd.Timestamp('2025-01-27')
        start_date = pd.Timestamp('2023-08-01')
    
    print(f"Downloading stock data from {start_date} to {end_date}...")
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval='1d')
    
    # Convert index to timezone-naive
    df.index = df.index.tz_localize(None)
    
    # Basic price and volume features
    df['Returns'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Price_Volatility'] = df['Close'].rolling(window=20).std()
    
    # Add technical indicators
    print("Calculating technical indicators...")
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['BB_high'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_low'] = ta.volatility.bollinger_lband(df['Close'])
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    
    # Get quarterly total liabilities
    print("Fetching balance sheet data...")
    balance_sheet = stock.quarterly_balance_sheet
    liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest']
    
    # Calculate liability changes with explicit fill method
    liabilities_df = pd.DataFrame(liabilities)
    liabilities_df.columns = ['Total_Liabilities']
    liabilities_df['Liability_Change'] = liabilities_df['Total_Liabilities'].pct_change(fill_method=None)
    
    # Ensure index is timezone-naive
    liabilities_df.index = pd.to_datetime(liabilities_df.index).tz_localize(None)
    
    # Create a daily date range and reindex liabilities
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    liabilities_df = liabilities_df.reindex(daily_dates, method='ffill')
    
    # Merge with daily data
    df = df.join(liabilities_df)
    
    # Forward fill and handle NaN values with explicit dtypes
    df['Total_Liabilities'] = pd.to_numeric(df['Total_Liabilities'], errors='coerce').ffill()
    df['Liability_Change'] = pd.to_numeric(df['Liability_Change'], errors='coerce').fillna(0)
    
    # Create a more volatile target based on market movements
    df['Market_Impact'] = (df['Returns'] * 0.3 + 
                          df['Volume_Change'] * 0.2 + 
                          df['Price_Volatility'] * 0.1)
    
    # Combine quarterly changes with daily market impact
    df['Daily_Liability_Change'] = (df['Liability_Change'] / 90) + df['Market_Impact']
    
    print(f"Using {len(df)} days of historical data from {df.index[0]} to {df.index[-1]}")
    
    return df.dropna()

def calculate_credit_rating(df, current_liabilities, ticker):
    """Calculate export finance-focused credit rating"""
    try:
        # Get market data
        market_data = yf.Ticker(ticker)
        
        # Get financial statements
        balance_sheet = market_data.balance_sheet.iloc[:, 0]  # Most recent
        income_stmt = market_data.income_stmt.iloc[:, 0]  # Most recent
        
        # Get key export-related metrics
        total_assets = balance_sheet.get('Total Assets', 0)
        current_assets = balance_sheet.get('Total Current Assets', 0)
        inventory = balance_sheet.get('Inventory', 0)
        accounts_receivable = balance_sheet.get('Net Receivables', 0)
        total_revenue = income_stmt.get('Total Revenue', 0)
        operating_income = income_stmt.get('Operating Income', 0)
        
        # Calculate export finance-specific ratios
        quick_ratio = (current_assets - inventory) / current_liabilities if current_liabilities else float('inf')
        receivables_turnover = total_revenue / accounts_receivable if accounts_receivable else float('inf')
        operating_margin = operating_income / total_revenue if total_revenue else 0
        working_capital_ratio = current_assets / current_liabilities if current_liabilities else float('inf')
        
        # Calculate market stability
        price_volatility = df['Returns'].std() * np.sqrt(252)  # Annualized volatility
        
        # Score each component (0-100)
        scores = {
            'Liquidity (Quick Ratio)': min(100, quick_ratio * 50),  # Higher weight for quick access to funds
            'Receivables Efficiency': min(100, receivables_turnover * 10),  # Important for trade finance
            'Operating Efficiency': min(100, operating_margin * 200),  # Ability to generate cash
            'Working Capital': min(100, working_capital_ratio * 40),  # Short-term financial health
            'Market Stability': max(0, 100 - (price_volatility * 100)),  # International market confidence
            'Asset Coverage': min(100, (total_assets / current_liabilities) * 30)  # Collateral strength
        }
        
        # Weighted average score with export finance focus
        weights = {
            'Liquidity (Quick Ratio)': 0.25,      # Critical for trade obligations
            'Receivables Efficiency': 0.20,        # Important for trade cycle
            'Operating Efficiency': 0.20,          # Cash generation ability
            'Working Capital': 0.15,               # Short-term financial strength
            'Market Stability': 0.10,              # International credibility
            'Asset Coverage': 0.10                 # Security for trade finance
        }
        
        final_score = sum(scores[k] * weights[k] for k in weights)
        
        # Export finance-adjusted rating scale
        rating_scale = {
            95: 'EF-AAA',  # Prime export finance rating
            90: 'EF-AA+',
            85: 'EF-AA',
            80: 'EF-AA-',
            75: 'EF-A+',
            70: 'EF-A',
            65: 'EF-A-',
            60: 'EF-BBB+',
            55: 'EF-BBB',  # Investment grade for trade finance
            50: 'EF-BBB-',
            45: 'EF-BB+',
            40: 'EF-BB',   # Speculative grade
            35: 'EF-BB-',
            30: 'EF-B+',
            25: 'EF-B',
            20: 'EF-B-',
            15: 'EF-CCC+', # High risk for trade finance
            10: 'EF-CCC',
            5: 'EF-CCC-',
            0: 'EF-D'      # Default level
        }
        
        rating = 'EF-D'
        for threshold in sorted(rating_scale.keys(), reverse=True):
            if final_score >= threshold:
                rating = rating_scale[threshold]
                break
        
        # Calculate 1-day forecast using Daily_Liability_Change instead of Daily_Change
        last_date = df.index[-1]
        last_value = current_liabilities
        
        # Use the daily liability change prediction model
        daily_change_pct = df['Daily_Liability_Change'].mean()  # Changed from Daily_Change
        volatility = df['Daily_Liability_Change'].std()  # Changed from Daily_Change
        
        # Add some randomness based on historical volatility
        forecast_change_pct = daily_change_pct + np.random.normal(0, volatility)
        forecast_value = last_value * (1 + forecast_change_pct)
        
        # Calculate absolute change
        absolute_change = forecast_value - last_value
        
        # Add to metrics dictionary
        scores['1-Day Forecast'] = min(100, max(0, 50 + (forecast_change_pct * 1000)))
        
        # Store forecast values for display
        forecast_info = {
            'absolute_change': absolute_change,
            'percentage_change': forecast_change_pct * 100
        }
        
        return {
            'rating': rating,
            'score': final_score,
            'metrics': scores,
            'forecast': forecast_info,
            'success': True
        }
    
    except Exception as e:
        print(f"Warning: Could not calculate export finance rating due to: {str(e)}")
        return {
            'rating': 'N/A',
            'score': 0,
            'metrics': {},
            'forecast': {'absolute_change': 0, 'percentage_change': 0},
            'success': False
        }

def calculate_trade_finance_cdo_metrics(df, current_liabilities, ticker):
    """Calculate CDO metrics for trade finance"""
    try:
        # Calculate predicted liabilities and their changes
        print(f"\nCalculating CDO metrics for {ticker}")
        
        # Create feature set for prediction
        features_df = create_features(ticker)
        if len(features_df) == 0:
            raise ValueError(f"No feature data available for {ticker}")
            
        print(f"Generated {len(features_df)} days of feature data")
        
        # Calculate daily changes from the features
        daily_changes = features_df['Returns'].values  # Using returns as proxy for liability changes
        volatility = np.std(daily_changes) * np.sqrt(252)  # Annualized volatility
        
        print(f"Calculated volatility: {volatility:.2%}")
        
        # Updated base PDs with more realistic values based on credit ratings
        base_pds = {
            'AAPL': 0.0003,  # 0.03% (AA+)
            'MSFT': 0.0001,  # 0.01% (AAA)
            'NVDA': 0.0015   # 0.15% (A)
        }
        pd_rate = base_pds.get(ticker, 0.0200)
        
        # Calculate recovery rate (using industry standards)
        base_recovery = 0.76  # Industry standard 76% recovery rate
        company_adjustment = {
            'AAPL': 0.05,  # Better than average due to strong balance sheet
            'MSFT': 0.05,  # Better than average due to strong balance sheet
            'NVDA': 0.03   # Slightly better due to market position
        }.get(ticker, 0.00)
        
        recovery_rate = min(0.95, base_recovery + company_adjustment)
        
        # Calculate LGD and expected loss
        loss_given_default = 1 - recovery_rate
        expected_loss = pd_rate * loss_given_default
        
        print(f"Calculated metrics for {ticker}:")
        print(f"PD Rate: {pd_rate:.4%}")
        print(f"Recovery Rate: {recovery_rate:.2%}")
        print(f"LGD: {loss_given_default:.2%}")
        print(f"Expected Loss: {expected_loss:.4%}")
        
        # Calculate USD metrics
        usd_metrics = {
            'Default_Risk': pd_rate,
            'Recovery_Rate': recovery_rate,
            'LGD': loss_given_default,
            'Expected_Loss': expected_loss,
            'Volatility': volatility
        }
        
        # Create display text
        usd_display = (
            f"USD Trade Finance Metrics\n"
            f"------------------------\n"
            f"Default Risk (1Y): {pd_rate:.4%}\n"
            f"Recovery Rate: {recovery_rate:.2%}\n"
            f"Loss Given Default: {loss_given_default:.2%}\n"
            f"Expected Loss: {expected_loss:.4%}\n"
            f"Volatility: {volatility:.2%}\n"
        )
        
        return {
            'metrics': usd_metrics,
            'display_text': usd_display,
            'success': True
        }
    
    except Exception as e:
        print(f"Warning: Could not calculate CDO metrics due to: {str(e)}")
        return {
            'metrics': {},
            'display_text': "Error calculating CDO metrics",
            'success': False
        }

def calculate_zar_trade_finance_metrics(df, current_liabilities, ticker):
    """Calculate South African (ZAR) trade finance metrics"""
    try:
        # ZAR specific parameters
        za_parameters = {
            'base_rate': 0.0850,      # South Africa repo rate
            'country_premium': 0.0300, # ZAR market premium
            'fx_volatility': 0.15,     # ZAR/USD volatility
            'spread_multiplier': 1.4   # South African market spread multiplier
        }
        
        # Get base metrics from USD calculation
        usd_metrics = calculate_trade_finance_cdo_metrics(df, current_liabilities, ticker)
        
        if not usd_metrics['success']:
            raise ValueError("Failed to calculate base USD metrics")
        
        # Adjust metrics for ZAR
        zar_metrics = {
            'Currency': 'ZAR',
            'Default_Risk': usd_metrics['metrics']['Default_Risk'] * za_parameters['spread_multiplier'],
            'Recovery_Rate': usd_metrics['metrics']['Recovery_Rate'] * 0.9,  # Adjust for ZAR market
            'LGD': usd_metrics['metrics']['LGD'] * 1.1,  # Higher LGD in ZAR
            'FX_Risk': za_parameters['fx_volatility'],
            'Total_Spread': (za_parameters['base_rate'] + 
                           za_parameters['country_premium'] + 
                           usd_metrics['metrics']['Expected_Loss'] * za_parameters['spread_multiplier'])
        }
        
        # Create display text for ZAR metrics
        zar_display = (
            f"South African Trade Finance Metrics\n"
            f"----------------------------------\n"
            f"Base Rate: {za_parameters['base_rate']:.2%}\n"
            f"Country Premium: {za_parameters['country_premium']:.2%}\n"
            f"FX Risk: {za_parameters['fx_volatility']:.1%}\n"
            f"Default Risk (ZAR): {zar_metrics['Default_Risk']:.4%}\n"
            f"Recovery Rate (ZAR): {zar_metrics['Recovery_Rate']:.2%}\n"
            f"Total Spread: {zar_metrics['Total_Spread']:.2%}"
        )
        
        return {
            'metrics': zar_metrics,
            'display_text': zar_display,
            'success': True
        }
    
    except Exception as e:
        print(f"Warning: Could not calculate ZAR metrics due to: {str(e)}")
        return {
            'metrics': {},
            'display_text': "Error calculating ZAR metrics",
            'success': False
        }

def calculate_tranche_pd(base_pd, attachment, detachment, spread_multiplier):
    """Calculate probability of default for specific tranche"""
    # Simple model: PD increases with attachment point and spread multiplier
    tranche_thickness = detachment - attachment
    relative_position = (1 - attachment) / tranche_thickness
    return base_pd * spread_multiplier * relative_position

def train_predict_liabilities(ticker='MSFT', future_months=3):
    """Train XGBoost model and predict daily liabilities"""
    # Ensure output directory exists
    output_dir = "data/predictions"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")
    
    # Set fixed dates
    current_date = pd.Timestamp('2025-01-27')
    display_start = pd.Timestamp('2024-08-01')
    forecast_end = pd.Timestamp('2025-04-30')
    training_start = pd.Timestamp('2023-08-01')
    
    print(f"Training period: {training_start} to {current_date}")
    print(f"Display period: {display_start} to {forecast_end}")
    
    # Get training data
    df = create_features(ticker, start_date=training_start, end_date=current_date)
    
    feature_columns = ['Returns', 'Volume_Change', 'Price_Volatility',
                      'SMA_20', 'SMA_50', 'BB_high', 'BB_low', 
                      'MFI', 'RSI', 'MACD']
    
    X = df[feature_columns].copy()
    y = df['Daily_Liability_Change'].copy()
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\nTraining XGBoost model...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.005,
        max_depth=5,
        min_child_weight=4,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_scaled, y)
    
    # Analyze and visualize feature importance
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    for idx, row in importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
        # Add visual bar for importance
        bar = "█" * int(row['importance'] * 100)
        print(f"  {bar}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance['feature'], importance['importance'])
    plt.title('Feature Importance in Liability Prediction')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    # Save feature importance plot
    importance_plot_path = os.path.join(output_dir, f"{ticker}_feature_importance.png")
    plt.savefig(importance_plot_path)
    plt.close()
    
    # Calculate correlation between features and target
    correlations = pd.DataFrame({
        'feature': feature_columns,
        'correlation': [abs(df[f].corr(df['Daily_Liability_Change'])) for f in feature_columns]
    }).sort_values('correlation', ascending=False)
    
    print("\nFeature Correlations with Daily Changes:")
    for idx, row in correlations.iterrows():
        print(f"{row['feature']}: {row['correlation']:.4f}")
    
    # Select top features (importance > 5%)
    top_features = importance[importance['importance'] > 0.05]['feature'].tolist()
    print(f"\nUsing top features: {top_features}")
    
    # Use only top features for predictions
    X = df[top_features].copy()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    # Make historical predictions
    predicted_changes = model.predict(X_scaled)
    
    # Get the anchor point (last known actual liability)
    anchor_date = df['Total_Liabilities'].last_valid_index()
    anchor_value = df.loc[anchor_date, 'Total_Liabilities']
    print(f"\nAnchor point: {anchor_date}: ${anchor_value:,.2f}")
    
    # Function to calculate daily liability values from an anchor point
    def calculate_daily_liabilities(df, anchor_date, anchor_value, forward=True):
        values = pd.Series(index=df.index, dtype=float)
        values.loc[anchor_date] = anchor_value
        
        # Get the pattern from Daily_Change
        pattern = df['Daily_Change']
        # Normalize pattern around 1
        normalized_pattern = (pattern - pattern.mean()) / pattern.std()
        # Scale pattern to small percentage changes (e.g., ±0.5%)
        scaled_pattern = normalized_pattern * 0.005  # 0.5% maximum daily change
        
        if forward:
            # Calculate forward from anchor following exact pattern
            current_value = anchor_value
            for date in df.index[df.index > anchor_date]:
                # Use pattern value directly
                pct_change = scaled_pattern.loc[date]
                current_value = current_value * (1 + pct_change)
                values.loc[date] = current_value
        else:
            # Calculate backward from anchor
            current_value = anchor_value
            for date in reversed(df.index[df.index < anchor_date]):
                pct_change = scaled_pattern.loc[date]
                current_value = current_value / (1 + pct_change)
                values.loc[date] = current_value
        
        return values
    
    # Calculate historical values
    historical_df = pd.DataFrame({
        'Actual_Total_Liabilities': df['Total_Liabilities'],
        'Daily_Change': predicted_changes * 100,
        'Is_Future': False
    }, index=df.index)
    
    # Calculate historical values before and after anchor
    historical_before = calculate_daily_liabilities(historical_df[historical_df.index <= anchor_date], 
                                                  anchor_date, anchor_value, forward=False)
    historical_after = calculate_daily_liabilities(historical_df[historical_df.index >= anchor_date], 
                                                 anchor_date, anchor_value, forward=True)
    
    historical_df['Predicted_Liabilities'] = pd.concat([historical_before, historical_after[1:]])
    
    # Generate future features
    print("\nGenerating future features...")
    future_dates = pd.date_range(
        start=current_date + pd.Timedelta(days=1),
        end=forecast_end,
        freq='D'
    )
    
    # Create future features based on recent patterns (only top features)
    future_features = pd.DataFrame(index=future_dates, columns=top_features)
    window = df.iloc[-30:].copy()
    
    future_changes = []
    for future_date in future_dates:
        # Update only the top features
        for feature in top_features:
            if feature in ['Returns', 'Volume_Change', 'Price_Volatility']:
                future_features.loc[future_date, feature] = window[feature].mean()
            elif feature in ['SMA_20', 'SMA_50']:
                future_features.loc[future_date, feature] = window['Close'].tail(int(feature.split('_')[1])).mean()
            elif feature in ['BB_high', 'BB_low']:
                std = window['Close'].std()
                mean = window['Close'].mean()
                future_features.loc[future_date, feature] = mean + (2 * std) if 'high' in feature else mean - (2 * std)
            else:  # RSI, MACD, MFI
                future_features.loc[future_date, feature] = window[feature].mean()
        
        # Scale features
        X_future = scaler.transform(future_features.loc[future_date:future_date])
        
        # Predict using XGBoost
        base_prediction = model.predict(X_future)[0]
        
        # Add controlled randomness based on model uncertainty
        prediction_std = predicted_changes[-30:].std()
        random_factor = np.random.normal(0, prediction_std * 0.3)
        final_prediction = base_prediction + random_factor
        
        future_changes.append(final_prediction)
        
        # Update rolling window
        new_row = df.iloc[-1].copy()
        new_row.name = future_date
        window = pd.concat([window[1:], pd.DataFrame([new_row])])
    
    # Calculate future values using same pattern-based approach
    future_df = pd.DataFrame({
        'Actual_Total_Liabilities': np.nan,
        'Daily_Change': np.array(future_changes) * 100,
        'Is_Future': True
    }, index=future_dates)
    
    last_historical_value = historical_df['Predicted_Liabilities'].iloc[-1]
    future_values = calculate_daily_liabilities(future_df, future_dates[0], last_historical_value, forward=True)
    future_df['Predicted_Liabilities'] = future_values
    
    # Combine and filter results
    result_df = pd.concat([historical_df, future_df], axis=0)
    mask = (result_df.index >= display_start) & (result_df.index <= forecast_end)
    result_df = result_df[mask].copy()
    
    # Calculate confidence intervals with 75% confidence level and tighter scale
    def calculate_confidence_intervals(df, forecast_mask, confidence=0.75):
        historical_std = df[~forecast_mask]['Daily_Change'].std() * 0.005  # Scale factor 0.005
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        forecast_values = df[forecast_mask]['Predicted_Liabilities']
        margin = z_score * historical_std * forecast_values
        
        upper = forecast_values + margin
        lower = forecast_values - margin
        
        return lower, upper
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot historical values
    historical_mask = ~result_df['Is_Future']
    plt.plot(result_df[historical_mask].index, 
            result_df[historical_mask]['Predicted_Liabilities'],
            color='green', label='Historical Predicted', alpha=0.7)
    
    # Plot forecast values
    forecast_mask = result_df['Is_Future']
    plt.plot(result_df[forecast_mask].index,
            result_df[forecast_mask]['Predicted_Liabilities'],
            color='blue', linestyle='--', label='Forecast', alpha=0.7)
    
    # Add 75% confidence intervals
    lower, upper = calculate_confidence_intervals(result_df, forecast_mask)
    plt.fill_between(result_df[forecast_mask].index,
                    lower, upper,
                    color='blue', alpha=0.15,
                    label='75% Confidence Interval')
    
    # Find the last historical date (actual current date)
    last_historical_date = result_df[historical_mask].index[-1]
    
    # Calculate credit rating before plotting - use original df instead of filtered
    current_liabilities = result_df.loc[last_historical_date, 'Predicted_Liabilities']
    credit_analysis = calculate_credit_rating(df, current_liabilities, ticker)  # Changed to use df
    
    # Add credit rating information to plot
    if credit_analysis['success']:
        forecast = credit_analysis['forecast']
        forecast_direction = "+" if forecast['absolute_change'] >= 0 else ""
        
        credit_info = (
            f"{ticker} Credit Analysis\n"
            f"Rating: {credit_analysis['rating']}\n"
            f"Score: {credit_analysis['score']:.1f}/100\n\n"
            f"1-Day Forecast:\n"
            f"  {forecast_direction}${abs(forecast['absolute_change'])/1e9:.3f}B "
            f"({forecast_direction}{forecast['percentage_change']:.2f}%)\n\n"
            f"Key Metrics:\n"
            + "\n".join([f"{k}: {v:.1f}/100" for k, v in credit_analysis['metrics'].items()])
        )
        
        # Make the credit info box more prominent
        plt.text(0.02, 0.98, credit_info,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor='white',
                         alpha=0.9,
                         edgecolor='gray'),
                fontsize=10,
                zorder=10)
    else:
        print(f"Warning: Could not calculate credit rating for {ticker}")
    
    # Add vertical line for current date
    plt.axvline(x=last_historical_date, 
                color='gray', 
                linestyle=':', 
                label='Current Date',
                linewidth=3)
    
    # Add horizontal line for historical mean
    historical_mean = result_df[historical_mask]['Predicted_Liabilities'].mean()
    plt.axhline(y=historical_mean, 
                color='red', 
                linestyle=':', 
                label=f'Historical Mean (${historical_mean/1e9:.1f}B)',
                linewidth=3)  # 3x thicker
    
    # Add scatter point for last historical value
    last_value = result_df.loc[last_historical_date, 'Predicted_Liabilities']
    plt.scatter(last_historical_date, last_value, 
               color='red', s=100, zorder=5,
               label=f'Current Value (${last_value/1e9:.1f}B)')
    
    # In the visualization section, add CDO analysis
    cdo_analysis = calculate_trade_finance_cdo_metrics(df, current_liabilities, ticker)
    
    if cdo_analysis['success']:
        # Add CDO info box on the right side of the plot
        plt.text(0.98, 0.98, cdo_analysis['display_text'],
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor='lightyellow',
                         alpha=0.9,
                         edgecolor='gray'),
                fontsize=10,
                zorder=10)
    
    plt.title(f'Predicted Total Liabilities (USD)')
    plt.ylabel('Total Liabilities ($)')
    plt.grid(True)
    plt.legend()
    
    # Format y-axis to show billions
    def billions_formatter(x, pos):
        return f'${x/1e9:.1f}B'
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(billions_formatter))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot with explicit confirmation
    liability_plot_path = os.path.join(output_dir, f"{ticker}_liability_predictions_usd.png")
    plt.savefig(liability_plot_path)
    print(f"{ticker} graph saved to: {liability_plot_path}")
    plt.close()
    
    print(f"\nFinal display range: {result_df.index[0]} to {result_df.index[-1]}")
    print(f"Starting liability value: ${result_df['Predicted_Liabilities'].iloc[0]:,.2f}")
    print(f"Ending liability value: ${result_df['Predicted_Liabilities'].iloc[-1]:,.2f}")
    
    return result_df 

print(os.path.exists('data/predictions/NVDA_liability_predictions_usd.png'))

# 1. Import the module
# from src.data import liability_predictor

# 2. Generate and save both graphs
# AAPL
aapl_df = train_predict_liabilities('AAPL', future_months=3)

# NVDA 
nvda_df = train_predict_liabilities('NVDA', future_months=3)

# 3. Verify saved files
import os
aapl_path = 'data/predictions/AAPL_liability_predictions_usd.png'
nvda_path = 'data/predictions/NVDA_liability_predictions_usd.png'

print(f"AAPL graph saved: {os.path.exists(aapl_path)}")
print(f"NVDA graph saved: {os.path.exists(nvda_path)}")

def calculate_default_correlation(df, ticker1, ticker2):
    """Calculate default correlation between two companies using predicted data"""
    try:
        print(f"\nCalculating correlation between {ticker1}-{ticker2}")
        print(f"Using predicted data from {df.index[0]} to {df.index[-1]}")
        
        # Get returns from the predicted liabilities
        returns = df['Predicted_Liabilities'].pct_change().dropna()
        print(f"Calculated returns using {len(returns)} data points")
        
        # Calculate correlation using the predicted returns
        correlation = returns.autocorr()
        print(f"Calculated base correlation: {correlation:.4f}")
        
        # Get base PDs
        base_pds = {
            'AAPL': 0.0003,  # 0.03% based on AA+ rating
            'MSFT': 0.0507,  # 5.07% as specified
            'NVDA': 0.0150   # 1.50% based on A rating
        }
        
        pd1 = base_pds.get(ticker1, 0.0200)
        pd2 = base_pds.get(ticker2, 0.0200)
        
        print(f"Using PDs: {ticker1}={pd1:.4%}, {ticker2}={pd2:.4%}")
        
        # Convert to asset correlation
        asset_correlation = 0.75 * correlation
        
        def default_correlation(rho, pd1, pd2):
            from scipy.stats import norm, multivariate_normal
            k1 = norm.ppf(pd1)
            k2 = norm.ppf(pd2)
            mean = [0, 0]
            cov = [[1, rho], [rho, 1]]
            bvn = multivariate_normal(mean, cov)
            joint_default_prob = bvn.cdf([k1, k2])
            default_corr = (joint_default_prob - pd1 * pd2) / np.sqrt(pd1 * (1 - pd1) * pd2 * (1 - pd2))
            return float(default_corr)
        
        def_corr = default_correlation(asset_correlation, pd1, pd2)
        
        print(f"\nResults for {ticker1}-{ticker2}:")
        print(f"Base Correlation: {correlation:.4%}")
        print(f"Asset Correlation: {asset_correlation:.4%}")
        print(f"Default Correlation: {def_corr:.4%}")
        print(f"Data Points Used: {len(returns)}\n")
        
        return {
            'equity_correlation': float(correlation),
            'asset_correlation': float(asset_correlation),
            'default_correlation': float(def_corr),
            'success': True
        }
        
    except Exception as e:
        print(f"\nError calculating default correlation between {ticker1}-{ticker2}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        return {
            'equity_correlation': 0,
            'asset_correlation': 0,
            'default_correlation': 0,
            'success': False
        }

def calculate_pool_correlations(df):
    """Calculate correlation matrix for the CDO pool"""
    tickers = ['AAPL', 'MSFT', 'NVDA']
    correlations = {}
    
    for i, ticker1 in enumerate(tickers):
        for ticker2 in tickers[i+1:]:
            corr = calculate_default_correlation(df, ticker1, ticker2)
            if corr['success']:
                correlations[f'{ticker1}-{ticker2}'] = {
                    'Default Correlation': corr['default_correlation'],
                    'Asset Correlation': corr['asset_correlation'],
                    'Equity Correlation': corr['equity_correlation']
                }
    
    if not correlations:
        print("Warning: No correlations could be calculated")
        return {}
        
    return correlations 

def visualize_cdo_results(results):
    """Create visualizations for CDO valuation results"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Ensure output directory exists
        output_dir = "data/predictions"
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style - Change from seaborn to a default matplotlib style
        plt.style.use('default')  # Changed from 'seaborn' to 'default'
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. CDO Structure (Waterfall)
        ax1 = plt.subplot(221)
        waterfall_data = [
            ('Senior (99%)', results['tranches']['senior']['size']),
            ('Equity (1%)', results['tranches']['equity']['size'])
        ]
        
        colors = ['#2ecc71', '#e74c3c']
        bottom = 0
        for i, (name, size) in enumerate(waterfall_data):
            ax1.bar(0, size, bottom=bottom, color=colors[i], label=name)
            bottom += size
        
        ax1.set_title('CDO Structure')
        ax1.set_ylabel('USD')
        ax1.set_xticks([])
        ax1.legend()
        
        # Format y-axis to show millions
        def millions_formatter(x, pos):
            return f'${x/1e6:.1f}M'
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))
        
        # 2. Value Components
        ax2 = plt.subplot(222)
        value_data = {
            'Gross Value': results['value']['gross_value'],
            'Net Value': results['value']['net_value'],
            'Notional': results['portfolio']['notional']
        }
        
        ax2.bar(value_data.keys(), value_data.values(), color=['#3498db', '#2980b9', '#95a5a6'])
        ax2.set_title('Value Components')
        ax2.set_ylabel('USD')
        ax2.tick_params(axis='x', rotation=45)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))
        
        # 3. Expected Loss by Company
        ax3 = plt.subplot(223)
        companies = ['MSFT', 'AAPL', 'NVDA']
        pds = [0.0001, 0.0003, 0.0015]  # From our calculations
        lgds = [0.19, 0.19, 0.21]       # From our calculations
        expected_losses = [pd * lgd * results['portfolio']['exposure_per_company'] 
                         for pd, lgd in zip(pds, lgds)]
        
        ax3.bar(companies, expected_losses, color=['#9b59b6', '#3498db', '#1abc9c'])
        ax3.set_title('Expected Loss by Company')
        ax3.set_ylabel('USD')
        
        # Format y-axis to show thousands
        def thousands_formatter(x, pos):
            return f'${x/1e3:.1f}K'
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(thousands_formatter))
        
        # 4. Yield Components
        ax4 = plt.subplot(224)
        yield_components = {
            'Risk-free': results['rates']['risk_free'],
            'Country Premium': results['rates']['country_premium'],
            'FX Premium': results['rates']['fx_premium'],
            'Senior Spread': results['rates']['senior_spread']
        }
        
        # Create stacked bar for yield components
        bottom = 0
        colors = ['#1abc9c', '#f1c40f', '#e67e22', '#e74c3c']
        for (name, value), color in zip(yield_components.items(), colors):
            ax4.bar(0, value, bottom=bottom, label=name, color=color)
            bottom += value
        
        ax4.set_title('Yield Components')
        ax4.set_ylabel('Percentage')
        ax4.set_xticks([])
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format y-axis as percentage
        def percentage_formatter(x, pos):
            return f'{x*100:.1f}%'
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(percentage_formatter))
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cdo_analysis.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"CDO analysis visualization saved to: {output_dir}/cdo_analysis.png")
        
        return True
        
    except Exception as e:
        print(f"Error creating CDO visualizations: {str(e)}")
        return False

def calculate_cdo_value_core(notional=10_000_000, equity_pct=0.01):
    """Calculate CDO value using Lévy copula for more accurate tail risk modeling"""
    try:
        # Portfolio structure
        exposure_per_company = notional / 3
        equity_tranche = notional * equity_pct
        senior_tranche = notional * (1 - equity_pct)
        
        # Base rates and spreads
        risk_free_rate = 0.0525  # 5.25%
        country_premium = 0.0300  # 3.00%
        fx_premium = 0.0200      # 2.00%
        base_rate = risk_free_rate + country_premium + fx_premium
        senior_spread = 0.0250   # 2.50%
        total_yield = base_rate + senior_spread
        
        # Fees and costs
        transaction_costs = notional * 0.0050  # 0.50%
        annual_servicing = notional * 0.0035  # 0.35%
        
        # Initialize results dictionary
        results = {
            'portfolio': {
                'notional': notional,
                'exposure_per_company': exposure_per_company,
                'total_expected_loss': 0,
                'expected_loss_rate': 0
            },
            'tranches': {
                'equity': {
                    'size': equity_tranche,
                    'expected_loss': 0,
                    'value': 0
                },
                'senior': {
                    'size': senior_tranche,
                    'expected_loss': 0,
                    'value': 0
                }
            },
            'costs': {
                'transaction_costs': transaction_costs,
                'annual_servicing': annual_servicing
            },
            'rates': {
                'risk_free': risk_free_rate,
                'country_premium': country_premium,
                'fx_premium': fx_premium,
                'senior_spread': senior_spread,
                'total_yield': total_yield
            }
        }
        
        # Default parameters
        companies = ['MSFT', 'AAPL', 'NVDA']
        pds = [0.0001, 0.0003, 0.0015]  # PDs
        lgds = [0.19, 0.19, 0.21]       # LGDs
        
        def clayton_levy_copula(u, theta=2.5):
            """Clayton-Lévy copula for tail dependence"""
            from scipy.special import gamma
            return (sum(u**(-theta)) - len(u) + 1)**(-1/theta)
        
        def inverse_stable_cdf(p, alpha=1.5, beta=0.5):
            """Inverse stable distribution for heavy-tailed modeling"""
            from scipy.stats import levy_stable
            return levy_stable.ppf(p, alpha, beta)
        
        # Calculate joint default probability using Lévy copula
        def calculate_joint_losses(nsim=10000):
            import numpy as np
            
            # Generate correlated uniform variables
            np.random.seed(42)  # for reproducibility
            uniforms = np.random.uniform(0, 1, (nsim, len(companies)))
            
            # Transform to heavy-tailed distribution
            losses = np.zeros((nsim, len(companies)))
            for i in range(len(companies)):
                # Apply Clayton-Lévy copula
                u = uniforms[:, i]
                v = clayton_levy_copula(u.reshape(-1, 1))
                
                # Transform to loss distribution
                losses[:, i] = inverse_stable_cdf(v) * lgds[i] * exposure_per_company
            
            return losses
        
        # Calculate losses
        portfolio_losses = calculate_joint_losses()
        total_el = np.mean(portfolio_losses.sum(axis=1))
        
        # Update portfolio metrics
        results['portfolio']['total_expected_loss'] = float(total_el)
        results['portfolio']['expected_loss_rate'] = float(total_el / notional)
        
        # Calculate tranche losses with tail risk consideration
        def calculate_tranche_loss(losses, attachment, detachment):
            tranche_losses = np.maximum(0, 
                                      np.minimum(detachment - attachment,
                                               (losses - attachment * notional) / 
                                               (detachment - attachment)))
            return np.mean(tranche_losses) * notional
        
        # Calculate losses per tranche
        equity_loss = calculate_tranche_loss(portfolio_losses.sum(axis=1), 0, equity_pct)
        senior_loss = calculate_tranche_loss(portfolio_losses.sum(axis=1), equity_pct, 1)
        
        # Calculate present values
        discount_factor = 1 / (1 + total_yield)
        
        # Update tranche values
        results['tranches']['equity']['expected_loss'] = float(equity_loss)
        results['tranches']['equity']['value'] = float((equity_tranche - equity_loss) * discount_factor)
        
        results['tranches']['senior']['expected_loss'] = float(senior_loss)
        results['tranches']['senior']['value'] = float((senior_tranche - senior_loss) * discount_factor)
        
        # Calculate total value
        results['value'] = {
            'gross_value': float(results['tranches']['equity']['value'] + 
                               results['tranches']['senior']['value']),
            'net_value': float(results['tranches']['equity']['value'] + 
                             results['tranches']['senior']['value'] - 
                             transaction_costs - 
                             annual_servicing)
        }
        
        # Print summary
        print("\nCDO Valuation Summary (USD) - Lévy Copula Model")
        print("=" * 50)
        print(f"Gross Portfolio Value: ${results['value']['gross_value']:,.2f}")
        print(f"Total Costs: ${transaction_costs + annual_servicing:,.2f}")
        print(f"Net CDO Value: ${results['value']['net_value']:,.2f}")
        print("\nTranche Details:")
        print(f"Equity Tranche (1%): ${results['tranches']['equity']['value']:,.2f}")
        print(f"Senior Tranche (99%): ${results['tranches']['senior']['value']:,.2f}")
        print(f"\nExpected Loss Rate: {results['portfolio']['expected_loss_rate']:.4%}")
        print(f"Total Yield: {total_yield:.2%}")
        
        return results
        
    except Exception as e:
        print(f"Error calculating CDO value with Lévy copula: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# This function will handle both calculation and visualization
def calculate_cdo_value(notional=10_000_000, equity_pct=0.01):
    """Calculate CDO value and visualize results"""
    results = calculate_cdo_value_core(notional, equity_pct)
    if results:
        visualize_cdo_results(results)
    return results

# Test the CDO visualization
cdo_results = calculate_cdo_value(notional=10_000_000, equity_pct=0.01) 