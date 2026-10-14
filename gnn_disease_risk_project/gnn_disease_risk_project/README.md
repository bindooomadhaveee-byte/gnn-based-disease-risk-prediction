# GNN-Based Disease Risk Prediction

Each patient is a node; similar patients are connected (kNN graph).
A Graph Convolutional Network (GCN) predicts disease risk (high/low) and is
compared against an MLP baseline.

## Setup
    pip install -r requirements.txt

## Run
    python gnn_disease_risk.py

## Use real data
In `main()`, uncomment:
    X, y = load_csv_data("diabetes.csv", label_col="Outcome")
(all feature columns must be numeric)

## Output
Accuracy, F1 and AUC for GCN and MLP, plus sample patient risk scores.
