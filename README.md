# Iris Flower Classification

A machine learning project that classifies Iris flowers into three species — *setosa*, *versicolor*, and *virginica* — based on petal and sepal measurements. Three classifiers are trained, compared, and evaluated.

## Results

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| K-Nearest Neighbors | ~96% | ~97% |
| Random Forest | ~96% | ~97% |
| Support Vector Machine | ~97% | ~97% |

All three models achieve high accuracy on this well-known benchmark dataset. SVM and Random Forest are the strongest performers.

## Project Structure

```
iris-flower-classification/
│
├── iris_classification.py   # Main script
├── iris_eda_distributions.png
├── iris_pairplot.png
├── iris_correlation.png
├── iris_confusion_matrix.png
├── iris_model_comparison.png
├── iris_feature_importance.png
└── README.md
```

## What the Script Does

1. **Loads the Iris dataset** directly from `sklearn.datasets` — no CSV download needed
2. **Exploratory Data Analysis (EDA)** — feature distribution histograms, pairplot, correlation heatmap
3. **Preprocessing** — train/test split (80/20), feature scaling with `StandardScaler`
4. **Model Training** — KNN, Random Forest, SVM trained and evaluated with 5-fold cross-validation
5. **Evaluation** — confusion matrix, classification report, model comparison bar chart
6. **Feature Importance** — Random Forest feature importance scores (petal length/width dominate)

## Key Finding

Petal length and petal width are by far the most discriminating features. Setosa is perfectly separable from the other two species; versicolor and virginica have slight overlap in petal measurements.

## How to Run

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Run the script
python iris_classification.py
```

All output charts are saved as PNG files in the same directory.

## Tech Stack

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (KNeighborsClassifier, RandomForestClassifier, SVC)

## Author

**Fahim Ahmed**  
2nd Year Student, Urban & Regional Planning  
Bangladesh University of Engineering and Technology (BUET)  
[LinkedIn](https://www.linkedin.com/in/fahim-ahmed-585b26357) | [GitHub](https://github.com/fmad121581-hub)
