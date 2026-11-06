# CART-Based Credit Risk Classification

**Student:** Pratik Pattewar  
**Student ID:** 2300030969  
**Programme:** B.Tech – Computer Science and Engineering  
**Institution:** KL University

## Project overview
This project demonstrates a Classification and Regression Tree (CART) model for binary credit-risk classification. It uses the public **Statlog (German Credit Data)** dataset rather than the synthetic loan-approval data in the supplied reference archive.

The implementation covers exploratory data analysis, Gini impurity, candidate binary split scoring, preprocessing, CART training, prediction, evaluation, decision-tree visualization and feature importance.

## Dataset
- **Name:** Statlog (German Credit Data)
- **Source:** UCI Machine Learning Repository
- **Records:** 1,000
- **Predictors:** 20
- **Target:** Credit risk (`1 = Good`, `2 = Bad` in the original source)
- **License:** CC BY 4.0
- **Citation:** Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77
- **Official page:** https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

### Dataset availability note
The execution environment used to assemble this ZIP did not permit downloading the dataset binary. To avoid fabricating or misrepresenting data, the ZIP contains a reproducible downloader rather than a fake CSV. Run `python download_dataset.py` once while connected to the internet. It retrieves the official UCI dataset through `ucimlrepo` and creates `data/german_credit.csv`. The notebook can also fetch the dataset automatically if the CSV is absent.

## Files
```text
Pratik_Pattewar_CART_Credit_Risk/
├── CART_Credit_Risk.ipynb
├── README.md
├── download_dataset.py
├── requirements.txt
└── data/
    └── (german_credit.csv is created by the downloader)
```

## Setup
Python 3.9 or later is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the dataset:
```bash
python download_dataset.py
```

Launch Jupyter:
```bash
jupyter notebook CART_Credit_Risk.ipynb
```

Run notebook cells from top to bottom. If `data/german_credit.csv` is absent, the notebook attempts to download the data using `ucimlrepo`.

## Methodology
1. Import the required Python libraries.
2. Load the UCI German Credit dataset.
3. Inspect data shape, data types, missing values and target distribution.
4. Explore numerical and categorical variables.
5. Separate predictors and target.
6. Calculate root Gini impurity manually.
7. Evaluate candidate numeric binary splits using weighted child Gini and Gini reduction.
8. Create a stratified 80:20 training/testing split.
9. One-hot encode categorical columns through a `ColumnTransformer`.
10. Train a CART classifier using `criterion="gini"`.
11. Generate predictions and evaluate performance.
12. Display a confusion matrix, decision tree and feature importance.
13. Demonstrate a prediction for an example applicant.

## Important interpretation notes
- The manual split table demonstrates numeric threshold splits at the root; the trained CART estimator performs its own recursive search on the preprocessed feature space.
- The model uses a fixed random seed to make the train/test split reproducible.
- Accuracy alone is insufficient for imbalanced credit-risk data. Review precision, recall, F1-score and the confusion matrix.
- UCI documents a cost matrix in which predicting a bad-risk applicant as good is more costly than the reverse. This notebook reports standard classification metrics and does not claim to optimize that cost matrix.
- The dataset is historical and intended for educational experimentation, not real lending decisions.

## Academic integrity and reproducibility
This notebook is an independent implementation based on a public dataset and a documented CART workflow. Review every section, run the cells yourself, and be prepared to explain the data preparation, Gini calculations, model parameters and evaluation results. Cite the original dataset when reusing it.
