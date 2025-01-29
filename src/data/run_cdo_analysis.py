from liability_predictor import (
    train_predict_liabilities,
    calculate_cdo_value
)

def main():
    # Generate liability predictions
    print("Generating AAPL predictions...")
    aapl_df = train_predict_liabilities('AAPL', future_months=3)
    
    print("\nGenerating NVDA predictions...")
    nvda_df = train_predict_liabilities('NVDA', future_months=3)
    
    print("\nGenerating MSFT predictions...")
    msft_df = train_predict_liabilities('MSFT', future_months=3)
    
    # Calculate CDO value
    print("\nCalculating CDO value...")
    cdo_results = calculate_cdo_value(notional=10_000_000, equity_pct=0.01)
    
    # Verify files
    import os
    aapl_path = 'data/predictions/AAPL_liability_predictions_usd.png'
    nvda_path = 'data/predictions/NVDA_liability_predictions_usd.png'
    msft_path = 'data/predictions/MSFT_liability_predictions_usd.png'
    cdo_path = 'data/predictions/cdo_analysis.png'
    
    print("\nVerifying saved files:")
    print(f"AAPL graph saved: {os.path.exists(aapl_path)}")
    print(f"NVDA graph saved: {os.path.exists(nvda_path)}")
    print(f"MSFT graph saved: {os.path.exists(msft_path)}")
    print(f"CDO analysis saved: {os.path.exists(cdo_path)}")

if __name__ == "__main__":
    main() 