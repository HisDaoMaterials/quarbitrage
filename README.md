![image](https://github.com/user-attachments/assets/9327d620-861d-41b4-bacf-263fb591d3c6)

A free-to-use Python library for retail investors and researchers to develop and test algorithmic trading strategies. This library implements statistical models and hypothesis tests inspired by academic articles and books in quantitative finance, enabling users to conduct rigorous analysis and backtesting. 

Note: This library is designed for use with Polars DataFrames, chosen for their superior speed and efficiency over Pandas. Compatibility with Pandas will be added in a future update.

### Version 0.6: Current status
- **quarbitrage/data_structures**:
  - standard_bars: Aggregate financial data at specified dollar/volume granularity. Bars exhibit improved statistical properties over raw financial data.
- **quarbitrage/data_engineering**:
  - labeling: Triple Barrier class labels
  - rolling_moments: Moving average, variance, exponentially-weighted moving averages / variances.
- **quarbitrage/preprocessing**:
  - datatypes: Convert string to datetime
- **quarbitrage/sampling**:
  - bootstrapping: Sequential bootstrap sampling, and overlap indicator matrix
- **quarbitrage/sql**: DatabaseClient API to read/write with local databases.
- **quarbitrage/utils**: Helper library

### Project Roadmap (1.0)
- **quarbitrage/ensemble**:
  - bagging: Complete sequential bootstrap ensemble model.
- **quarbitrage/statistics**: Implement financial statistics (Sharpe Ratio / Max Drawdown etc.) and hypothesis testing functions.
- **quarbitrage/feature_importance**: Implement feature selection tools for machine learning models.
- **quarbitrage/bet_sizing**: Implement optimal bet sizing functions (e.g., Kelly Criterion).

## Features
<ul>
  <li>
    <b>Statistical Modeling:</b> Machine learning models for financial data analysis and predictive modeling.
  </li>
  <li>
    <b>Hypothesis Testing:</b> Implementations of statistical tests for evaluating trading strategy metrics like Sharpe ratios, drawdowns.
  </li>
  <li>
    <b>Backtesting Framework:</b> Methods to simulate trading strategies and assess performance.
  </li>
</ul>

## Current Implementations
<ul>
  <li>
    Techniques from <i>Advances in Financial Machine Learning</i> (2018) by Marcos López de Prado.
  </li>
</ul>

## Installation

Awaiting Version 1.0 completion

## Getting Started

Awaiting Version 1.0 completion

## License
Quarbitrage is available under the MIT license. See the LICENSE file for more info.
## Disclaimer

This library is for educational and research purposes only. It does not constitute financial advice.




