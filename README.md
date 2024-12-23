# Transaction Fraud Detection Using GNNs and Tabular Models

This repository presents a comprehensive framework for fraud detection in financial transactions using both supervised and semi-supervised learning techniques. The dataset contains transaction details with labels indicating fraudulent or legitimate transactions.

## Features

- **Exploratory Data Analysis (EDA)**:
  - In-depth analysis of numerical and categorical features.
  - Temporal and categorical trend visualizations.
  - Identification of data imbalances and patterns.

- **Supervised Learning**:
  - Tabular models using engineered features.
  - Oversampling techniques (SMOTE) for handling class imbalance.
  - Comparison of models with and without feature engineering.

- **Semi-Supervised Learning**:
  - Graph-based models (Graph Attention Network (GAT) and GraphSAGE).
  - Hybrid loss combining classification and graph reconstruction objectives.
  - Alternative approaches: Autoencoders, Isolation Forest, Local Outlier Factor, and Gaussian Mixture Models.

- **Visualization**:
  - Temporal analysis of fraudulent transactions.
  - Network graph visualizations showing source-destination-agent relationships.
  - Distribution plots for transaction amounts and fraud probabilities.

## Dataset

The dataset consists of financial transaction records with the following columns:
- **`date`**: Transaction timestamp in milliseconds.
- **`user`**: Unique user identifier.
- **`source_prefix`**, **`source_postfix`**: Attributes related to the transaction's source.
- **`dest_prefix`**, **`dest_postfix`**: Attributes related to the transaction's destination.
- **`agent`**: Categorical variable indicating the handler of the transaction.
- **`amount`**: Transaction value in the smallest monetary unit.
- **`status`**: Indicates the transaction's result (e.g., success or fail).
- **`label`**: Binary variable indicating fraud (`1`) or non-fraud (`0`).

### Key Highlights:
- **106,036 rows**, **10 columns**.
- No missing values.
- Highly imbalanced, with ~3.5% fraudulent transactions.

## Methods

### 1. Exploratory Data Analysis (EDA)
- Insights into data distribution, trends, and relationships.
- Fraud vs. non-fraud analysis for key features like `amount`, `agent`, and `status`.

### 2. Supervised Learning
- **Models**: Tabular Transformers, Neural Networks.
- **Feature Engineering**:
  - Transaction-specific features: `is_high_risk_pair`, `fraud_rate`.
  - Temporal features: `is_night_high_risk`.
- **Oversampling**: SMOTE for handling class imbalance.
- **Evaluation Metrics**: Precision, Recall, F1-Score.

### 3. Semi-Supervised Learning
- **Graph-Based Models**:
  - Graph Attention Networks (GAT).
  - GraphSAGE.
  - Hybrid loss combining node classification and reconstruction.
- **Other Methods**:
  - Autoencoders.
  - Isolation Forest.
  - Local Outlier Factor.
  - Gaussian Mixture Models.

### 4. Visualization
- Temporal distribution of fraud cases.
- Network graphs of fraudulent transactions.
- Feature importance and correlation heatmaps.

## Results

| Method              | Precision | Recall | F1-Score |
|---------------------|-----------|--------|----------|
| Supervised (Engineered) | 0.87      | 0.84   | 0.85     |
| Supervised (SMOTE)      | 0.79      | 0.91   | 0.85     |
| GAT (15%labaled)                 | 0.82      | 0.88   | 0.85     |
| GraphSAGE (15%labaled)           | 0.80      | 0.85   | 0.82     |

> Note: Results may vary based on parameter tuning and dataset changes.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset Source**: Provided during an interview task.
- **Referenced Papers**:
  - [Semi-Supervised Learning on Graphs (Kipf et al.)](https://arxiv.org/abs/1609.02907)



