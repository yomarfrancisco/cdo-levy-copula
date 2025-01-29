# CDO Valuation using Lévy Copula

Advanced CDO (Collateralized Debt Obligation) valuation system using Lévy copula for improved tail risk modeling.

## Features

- 🔄 Dynamic liability prediction using XGBoost
- 📊 Lévy copula implementation for accurate tail dependence
- 💹 Export finance credit rating system
- 📈 Technical analysis integration
- 🎯 Advanced visualization of CDO metrics

## Mathematical Foundation

The model uses a Clayton-Lévy copula for tail dependence modeling:

```python
def clayton_levy_copula(u, theta=2.5):
    return (sum(u**(-theta)) - len(u) + 1)**(-1/theta)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cdo-levy-copula.git
cd cdo-levy-copula
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

```python
from src.data.levy_copula_cdo import calculate_cdo_value

# Calculate CDO value with Lévy copula
results = calculate_cdo_value(
    notional=10_000_000, 
    equity_pct=0.01
)
```

## Example Output

The model provides detailed CDO metrics including:
- Tranche-specific valuations
- Expected loss calculations
- Tail risk assessments
- Visual analytics

## License

MIT License - see LICENSE file for details
