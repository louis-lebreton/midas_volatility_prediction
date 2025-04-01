# Geopolitical Risk Uncertainty and Oil Future Volatility: Evidence from MIDAS Models

This repository contains the implementation and analysis of Mixed Data Sampling (MIDAS) models to study the relationship between geopolitical risk uncertainty and oil futures volatility. The project explores how different components of geopolitical risk indices affect the volatility forecasting of crude oil futures.

## Project Overview

The research examines the predictive power of various MIDAS models for oil market volatility, incorporating:
- Multiple realized volatility measures (RV, RS, CJ)
- Geopolitical Risk (GPR) indices and their decomposition
- Different forecasting horizons

## Key Features

- Decomposition of oil price volatility into continuous and jump components
- Decomposition of geopolitical risk indices into expected and shocked components
- Implementation of multiple MIDAS model variants
- Out-of-sample forecast evaluation using HMSE, HMAE metrics
- Model Confidence Set (MCS) tests for comparative model performance

## Prerequisites

This project requires Python 3.11 and several specific packages. All dependencies are listed in the `requirements.txt` file.

### Installation

1. Clone this repository
```bash
git clone https://github.com/louis-lebreton/midas_volatility_prediction.git
cd gpr-oil-volatility
```

2. Create a virtual environment and activate it
```bash
python -m venv .venv
# On Windows
.\.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. Install the required packages
```bash
pip install -r requirements.txt
```

## Directory Structure

```
├── data/
│   ├── raw/               # Raw data files (GPR indices, oil futures prices)
│   └── result/            # Generated volatility measures and results
├── src/
│   ├── volatility_measures.py  # Class for computing volatility measures
│   ├── gpr_measures.py         # Class for GPR index decomposition
│   ├── midas_models.py         # Class for implementation of MIDAS models
│   └── model_evaluation.py     # Class for model comparison and evaluation
├── midas_modelisation.ipynb    # Main notebook with complete analysis
└── requirements.txt            # Required Python packages
```
