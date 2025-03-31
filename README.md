# Neural Network Model for Alphabet Soup Fund Selection

## Overview

Alphabet Soup, a nonprofit foundation, seeks a predictive tool to identify the most promising applicants for funding. The goal is to develop a **binary classification model** that predicts whether an organization will be **successful** if funded. Using historical data of over **34,000** previously funded organizations, we aim to build a machine learning model that enhances Alphabet Soup’s funding decisions.

## Dataset Description

The dataset includes key metadata about past funding recipients:
- **EIN and NAME**: Organization identifiers.
- **APPLICATION_TYPE**: Type of funding application submitted.
- **AFFILIATION**: Sector classification of the organization.
- **CLASSIFICATION**: Government classification.
- **USE_CASE**: Purpose for which funding is requested.
- **ORGANIZATION**: Organization type (e.g., nonprofit, government, etc.).
- **STATUS**: Whether the organization is active.
- **INCOME_AMT**: Income category of the organization.
- **SPECIAL_CONSIDERATIONS**: Unique factors affecting applications.
- **ASK_AMT**: Amount of funding requested.
- **IS_SUCCESSFUL**: Target variable indicating whether the organization successfully utilized funding.

## Objective

Develop a **Neural Network Model** to predict **IS_SUCCESSFUL** based on the provided features. The model will help Alphabet Soup make data-driven funding decisions.

## Model Development Approach

### **1. Data Preprocessing**

- **Data Cleaning**:
  - No missing values in the dataset; hence, no imputation required.
  - Categorical features such as **APPLICATION_TYPE** and **CLASSIFICATION** are encoded using one-hot encoding.
  - Rare categories in **APPLICATION_TYPE** and **CLASSIFICATION** are consolidated into an "Other" category to reduce dimensionality.
  - **SPECIAL_CONSIDERATIONS** is label encoded (‘N’ → 0, ‘Y’ → 1).
  
- **Feature Scaling**:
  - StandardScaler is applied to numerical features to improve neural network performance.
  - One-hot encoding ensures categorical variables are appropriately transformed.
  
- **Data Splitting**:
  - The dataset is split into training and testing sets.

### **2. Neural Network Architecture**

The neural network model is structured as follows:

1. **Input Layer**: Number of input neurons equals the total preprocessed features.
2. **Hidden Layers**:
   - **Layer 1**: 80 neurons, **ReLU activation**.
   - **Layer 2**: 30 neurons, **Tanh activation**.
3. **Output Layer**: 1 neuron, **Sigmoid activation** (for binary classification).
4. **Loss Function**: **Binary Crossentropy**.
5. **Optimizer**: **Adam** (adaptive learning rate optimization).
6. **Evaluation Metric**: **Accuracy**.

### **3. Model Training**

- The model is trained using the **training dataset** (`X_train_scaled`, `y_train`).
- The model is trained for **100 epochs** to optimize learning.

### **4. Model Evaluation**

After training, model performance is evaluated using:

- **Accuracy**: Percentage of correctly predicted outcomes.

### **5. Results**

- **Train Metrics**: Evaluation metrics on the training set.
- **Test Metrics**: Model performance on unseen test data.

### **6. Conclusion and Future Recommendations**

#### **Model Optimization Strategies:**
1. **Hyperparameter Tuning**:
   - Experimenting with different numbers of layers and neurons.
   - Trying alternative activation functions and dropout layers to prevent overfitting.
2. **Additional Preprocessing**:
   - Addressing class imbalance using class weights or oversampling techniques.
3. **Alternative Models**:
   - Testing **Random Forest** or **XGBoost**, which often perform better on structured/tabular datasets.

### **Final Thoughts**

By refining this model, Alphabet Soup can **enhance decision-making**, ensuring that funding is allocated to organizations with the highest potential for success. Further iterations and comparative testing of different machine learning models can optimize predictions for future funding applicants.

