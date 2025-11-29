"""
US Treasury Curve Fetching and Processing

This module provides functionality to fetch current US Treasury yields
and construct a yield curve for model calibration.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
import json

# Try to import requests for live data fetching
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .hull_white_2f import YieldCurve


# Standard US Treasury maturities (in years)
TREASURY_MATURITIES = {
    '1M': 1/12,
    '2M': 2/12,
    '3M': 3/12,
    '4M': 4/12,
    '6M': 6/12,
    '1Y': 1.0,
    '2Y': 2.0,
    '3Y': 3.0,
    '5Y': 5.0,
    '7Y': 7.0,
    '10Y': 10.0,
    '20Y': 20.0,
    '30Y': 30.0
}


def fetch_treasury_rates() -> Optional[Dict[str, float]]:
    """
    Fetch current US Treasury rates from Treasury.gov API.
    
    Returns:
        Dictionary mapping tenor names to yields (in decimal form),
        or None if fetch fails.
    """
    if not REQUESTS_AVAILABLE:
        return None
    
    try:
        # Treasury.gov API for daily rates
        url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates"
        params = {
            "filter": "record_date:gte:2024-01-01,security_type_desc:eq:Treasury Bills,Treasury Notes,Treasury Bonds",
            "sort": "-record_date",
            "page[size]": 100
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Process and return rates
            rates = {}
            for record in data.get('data', []):
                desc = record.get('security_desc', '')
                rate = float(record.get('avg_interest_rate_amt', 0)) / 100
                # Map to our tenor format
                if '1-month' in desc.lower():
                    rates['1M'] = rate
                elif '3-month' in desc.lower():
                    rates['3M'] = rate
                # ... etc
            return rates if rates else None
    except Exception as e:
        print(f"Failed to fetch Treasury rates: {e}")
    
    return None


def get_default_treasury_rates() -> Dict[str, float]:
    """
    Get default/sample US Treasury rates.
    
    These are representative rates based on typical market conditions.
    Rates are in decimal form (e.g., 0.05 = 5%).
    
    Updated to reflect approximate Nov 2024 market rates.
    """
    return {
        '1M': 0.0535,   # 5.35%
        '2M': 0.0530,   # 5.30%
        '3M': 0.0525,   # 5.25%
        '4M': 0.0518,   # 5.18%
        '6M': 0.0505,   # 5.05%
        '1Y': 0.0475,   # 4.75%
        '2Y': 0.0435,   # 4.35%
        '3Y': 0.0420,   # 4.20%
        '5Y': 0.0415,   # 4.15%
        '7Y': 0.0420,   # 4.20%
        '10Y': 0.0430,  # 4.30%
        '20Y': 0.0465,  # 4.65%
        '30Y': 0.0450   # 4.50%
    }


def convert_to_continuous_rates(rates: Dict[str, float]) -> Dict[str, float]:
    """
    Convert semi-annual compounding yields to continuously compounded rates.
    
    Treasury yields are quoted on semi-annual bond equivalent basis.
    r_continuous = 2 * ln(1 + y_quoted/2)
    """
    continuous = {}
    for tenor, rate in rates.items():
        continuous[tenor] = 2 * np.log(1 + rate / 2)
    return continuous


def get_us_treasury_curve(use_live_data: bool = False) -> YieldCurve:
    """
    Construct a US Treasury yield curve.
    
    Args:
        use_live_data: If True, attempt to fetch live rates (requires internet)
        
    Returns:
        YieldCurve object calibrated to US Treasury rates
    """
    # Get rates
    if use_live_data:
        rates = fetch_treasury_rates()
        if rates is None:
            print("Live data fetch failed, using default rates")
            rates = get_default_treasury_rates()
    else:
        rates = get_default_treasury_rates()
    
    # Convert to continuous compounding
    continuous_rates = convert_to_continuous_rates(rates)
    
    # Build arrays for YieldCurve
    maturities = []
    yields = []
    
    for tenor, maturity in sorted(TREASURY_MATURITIES.items(), key=lambda x: x[1]):
        if tenor in continuous_rates:
            maturities.append(maturity)
            yields.append(continuous_rates[tenor])
    
    return YieldCurve(np.array(maturities), np.array(yields))


def get_model_parameters() -> dict:
    """
    Get recommended Two-Factor Hull-White parameters for US rates.
    
    These parameters are calibrated based on typical US interest rate dynamics:
    - Mean reversion 'a' around 0.1 implies ~7 year half-life for short rate
    - Mean reversion 'b' around 0.03-0.05 implies longer persistence for level factor
    - Volatility parameters calibrated to historical rate volatility
    - Negative correlation captures the tendency for yield curve to flatten 
      when rates rise
    """
    return {
        'a': 0.10,          # Short rate mean reversion
        'b': 0.04,          # Second factor mean reversion  
        'sigma1': 0.0100,   # Short rate volatility (100 bps/year)
        'sigma2': 0.0080,   # Second factor volatility (80 bps/year)
        'rho': -0.30        # Correlation between factors
    }


def get_parameter_descriptions() -> dict:
    """Get descriptions of model parameters."""
    return {
        'a': {
            'name': 'Short Rate Mean Reversion',
            'description': 'Speed at which short rate reverts to mean',
            'typical_range': [0.01, 0.5],
            'unit': '1/year'
        },
        'b': {
            'name': 'Second Factor Mean Reversion',
            'description': 'Speed at which second factor reverts to zero',
            'typical_range': [0.01, 0.2],
            'unit': '1/year'
        },
        'sigma1': {
            'name': 'Short Rate Volatility',
            'description': 'Instantaneous volatility of short rate',
            'typical_range': [0.005, 0.02],
            'unit': 'decimal (0.01 = 100 bps)'
        },
        'sigma2': {
            'name': 'Second Factor Volatility',
            'description': 'Instantaneous volatility of second factor',
            'typical_range': [0.003, 0.015],
            'unit': 'decimal'
        },
        'rho': {
            'name': 'Factor Correlation',
            'description': 'Correlation between the two Brownian motions',
            'typical_range': [-0.8, 0.3],
            'unit': 'dimensionless'
        }
    }

