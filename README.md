# Least Squares Linear System Solver

A Streamlit app to solve inconsistent linear systems **Ax = b** using the Least Squares method and compare it with sklearn's Linear Regression.

## Features

- Upload dataset (CSV with columns a1–a8 and b)
- Solve Ax = b manually using formula: x = (AᵀA)⁻¹Aᵀb
- Compare with sklearn LinearRegression
- MSE comparison, prediction plots, and coefficient charts

## Installation
```bash
pip install -r requirements.txt

## Usage

bash
streamlit run project2_least_squares.py

## Dataset Format

CSV file with columns: `a1, a2, a3, a4, a5, a6, a7, a8, b`

## Requirements

- streamlit
- numpy
- pandas
- matplotlib
- scikit-learn


---

