# Breast Cancer Classification

This repository contains a Jupyter notebook that demonstrates the process of training and evaluating various machine learning models to classify breast cancer data. The primary goal is to determine the best model based on several performance metrics, including accuracy, F1-score, precision, recall, and balanced accuracy.

## About the Dataset

### Description

Breast cancer is the most common cancer among women worldwide, accounting for 25% of all cancer cases. In 2015 alone, it affected over 2.1 million people. The disease starts when cells in the breast begin to grow uncontrollably, typically forming tumors that can be detected via X-ray or felt as lumps.

The key challenge in treating breast cancer is accurately classifying tumors as either malignant (cancerous) or benign (non-cancerous). This project focuses on using machine learning models to make this classification using the Breast Cancer Wisconsin (Diagnostic) Dataset.

### Acknowledgments

This dataset has been referred from Kaggle.

### Objective

1. **Understand the Dataset & Cleanup**: Explore and preprocess the dataset to prepare it for model training.
2. **Build Classification Models**: Implement and train multiple machine learning models to predict whether the cancer type is malignant or benign.
3. **Fine-Tune Hyperparameters**: Optimize the models by fine-tuning hyperparameters.
4. **Compare Evaluation Metrics**: Evaluate and compare the performance of various classification algorithms using key metrics.

## Dataset Features

The dataset contains the following columns:

| Column                  | Description                                   | Dtype  |
|-------------------------|-----------------------------------------------|--------|
| `id`                    | Identifier for each instance                  | int64  |
| `diagnosis`             | Target variable - Malignant (M) or Benign (B) | object |
| `radius_mean`           | Mean of distances from center to perimeter points | float64 |
| `texture_mean`          | Standard deviation of gray-scale values       | float64 |
| `perimeter_mean`        | Mean size of the tumor perimeter              | float64 |
| `area_mean`             | Mean size of the tumor area                   | float64 |
| `smoothness_mean`       | Local variation in radius lengths             | float64 |
| `compactness_mean`      | Perimeter^2 / area - 1.0                      | float64 |
| `concavity_mean`        | Severity of concave portions of the contour   | float64 |
| `concave points_mean`   | Number of concave portions of the contour     | float64 |
| `symmetry_mean`         | Symmetry of the tumor                         | float64 |
| `fractal_dimension_mean`| Fractal dimension ("coastline approximation") | float64 |
| `radius_se`             | Standard error for the mean radius            | float64 |
| `texture_se`            | Standard error for the mean texture           | float64 |
| `perimeter_se`          | Standard error for the mean perimeter         | float64 |
| `area_se`               | Standard error for the mean area              | float64 |
| `smoothness_se`         | Standard error for the mean smoothness        | float64 |
| `compactness_se`        | Standard error for the mean compactness       | float64 |
| `concavity_se`          | Standard error for the mean concavity         | float64 |
| `concave points_se`     | Standard error for the mean concave points    | float64 |
| `symmetry_se`           | Standard error for the mean symmetry          | float64 |
| `fractal_dimension_se`  | Standard error for the mean fractal dimension | float64 |
| `radius_worst`          | Worst (largest) value for the mean radius     | float64 |
| `texture_worst`         | Worst (largest) value for the mean texture    | float64 |
| `perimeter_worst`       | Worst (largest) value for the mean perimeter  | float64 |
| `area_worst`            | Worst (largest) value for the mean area       | float64 |
| `smoothness_worst`      | Worst (largest) value for the mean smoothness | float64 |
| `compactness_worst`     | Worst (largest) value for the mean compactness| float64 |
| `concavity_worst`       | Worst (largest) value for the mean concavity  | float64 |
| `concave points_worst`  | Worst (largest) value for the mean concave points | float64 |
| `symmetry_worst`        | Worst (largest) value for the mean symmetry   | float64 |
| `fractal_dimension_worst`| Worst (largest) value for the mean fractal dimension | float64 |

### Features Used for Analysis

After analyzing the correlation with the target variable `diagnosis`, the following features were selected for model training and evaluation:

- `radius_mean`
- `texture_mean`
- `perimeter_mean`
- `area_mean`
- `smoothness_mean`
- `compactness_mean`
- `concavity_mean`
- `concave points_mean`
- `symmetry_mean`
- `radius_se`
- `perimeter_se`
- `area_se`
- `compactness_se`
- `concavity_se`
- `concave points_se`
- `radius_worst`
- `texture_worst`
- `perimeter_worst`
- `area_worst`
- `smoothness_worst`
- `compactness_worst`
- `concavity_worst`
- `concave points_worst`
- `symmetry_worst`
- `fractal_dimension_worst`

## Models Evaluated

The following machine learning models were implemented and evaluated in the notebook:

1. **Logistic Regression**
2. **Decision Tree**
3. **Naive Bayes**
4. **Random Forest**
5. **K-Nearest Neighbors (KNN)**
6. **Support Vector Machine (SVM)**
7. **XGBoost**
8. **Multi-Layer Perceptron (MLP)**
9. **Custom Neural Network** (built using Keras)

## Evaluation Metrics

The models were evaluated based on the following metrics:

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **F1-Score**: The harmonic mean of precision and recall, useful for imbalanced datasets.
- **Precision**: The ratio of true positive predictions to the total positive predictions.
- **Recall**: The ratio of true positive predictions to the total actual positives.
- **Balanced Accuracy**: The average of recall obtained on each class, adjusting for class imbalance.

## Key Findings

- **Neural Networks** (both MLP and custom) outperformed other models, achieving the highest accuracy (~98.2%), F1-score (~97.6%), precision (1.0), and recall (~95.3%).
- **SVM** and **Logistic Regression** were also strong performers, particularly in accuracy and F1-score.
- **Decision Tree** had lower precision compared to others, indicating a tendency towards more false positives, despite a decent recall.
- **Random Forest** and **XGBoost** offered a good balance between accuracy and computational efficiency.

## Conclusion

The **Neural Networks** are recommended for scenarios where performance is paramount. **SVM** and **Logistic Regression** are suitable alternatives when simpler, more interpretable models are preferred. **Decision Tree** and **Naive Bayes** might be useful in certain contexts but were generally outperformed by the more complex models.

## Visualizations

The notebook includes visualizations that compare the models across different metrics, making it easy to identify the best-performing models. These include bar charts with distinct color scales for each metric, providing a clear visual comparison.

## How to Use This Notebook

1. **Clone the Repository**: Clone the repository to your local machine using:
   ```bash
   git clone https://github.com/HAFDAOUIH/Breast-Cancer-Classification.git
