# Practice 4 - Boston Housing Dataset Analysis

This project demonstrates the analysis and modeling of the Boston Housing dataset using multiple regression models, including a custom-built Multilayer Perceptron (MLP) implemented in PyTorch. The analysis also includes performance comparisons with Linear Regression, Decision Tree Regressor, and Random Forest Regressor.

---

## Installation
Install the required libraries:
```bash
python3 -m pip install -r requirements.txt
```

---

## Jupyter Notebook

The notebook contains the following key sections:

### 1. Data Loading and Preprocessing
- The Boston Housing dataset is loaded using `sklearn.datasets`.
- Data is split into features (X) and target variable (y).
- StandardScaler is applied to scale the features.

### 2. Multilayer Perceptron (MLP) Model
- A PyTorch-based MLP model is defined with fully connected layers and ReLU activation functions.
- The model is trained over 100 epochs using the Adam optimizer and Mean Squared Error (MSE) loss function.

### 3. Comparison with Other Models
The following models are implemented and compared:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

### 4. Performance Evaluation
- Models are evaluated using Mean Squared Error (MSE) and R-squared metrics.
- Results are visualized through bar charts and scatter plots.

### 5. Visualizations
- **Bar Charts**: Comparing MSE and R-squared scores for all models.
- **Scatter Plots**: Comparing actual vs. predicted values for each model.

---

## Results

### Performance Metrics:
- Mean Squared Error (MSE) and R-squared values are calculated for all models.
- Scatter plots visualize the differences in predictions for each model.

### Key Insights:
- The MLP model is trained to predict housing prices with competitive performance compared to traditional regression models.
- The results showcase the strengths and weaknesses of each regression approach.


