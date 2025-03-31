# **Neural Network Model for Alphabet Soup Fund Selection: Performance Analysis**

## **Overview of the Analysis:**
The purpose of this analysis is to evaluate the performance of a deep learning model designed to predict whether an organization receiving funding from the Alphabet Soup nonprofit foundation will successfully use the funds. Using a binary classification approach, we aim to predict success (`1`) or failure (`0`) for funding requests based on several input features from a dataset containing over 34,000 historical records.

The dataset includes various metadata for each organization, such as application type, classification, funding request, and the success of previous funding. The goal of this analysis is to build and evaluate a deep neural network model that can predict future success or failure in funding applications.

## **Results:**

#### **Data Preprocessing:**
Before feeding the data into the neural network model, several preprocessing steps were undertaken to ensure the dataset was suitable for machine learning.

- **Target Variable:**
  - The target variable (`y`) is `IS_SUCCESSFUL`, which indicates whether the funding was successfully used by the organization (1 for successful, 0 for unsuccessful).
  
- **Feature Variables:**
  - The feature variables (`X`) are the rest of the columns in the dataset, including metadata such as:
    - `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`, and others that describe the organization and its funding request.
    
- **Variables to Remove:**
  - The columns `EIN` and `NAME` were removed from the input data since they are identifiers and do not contribute to predicting the success of funding.

- **Data Cleaning:**
  - Categorical variables such as `APPLICATION_TYPE` and `CLASSIFICATION` were encoded using one-hot encoding and special labels (e.g., replacing rare categories with "Other").
  - The target variable `SPECIAL_CONSIDERATIONS` was ordinal encoded to map `"N"` to `0` and `"Y"` to `1`.

- **Scaling the Data:**
  - The features were scaled using `StandardScaler` to ensure that the input values are on the same scale, which improves the performance of neural networks.

#### **Compiling, Training, and Evaluating the Model:**

- **Model Architecture:**
  - **Neurons and Layers:**
    - **Input Layer:** The model has 42 input features (`X_train.shape[1]`).
    - **Hidden Layers:** 
      - The first hidden layer contains **80 neurons**, and the second hidden layer has **30 neurons**. These values were chosen based on typical configurations in practice, balancing model complexity with computational cost.
    - **Output Layer:** The output layer consists of **1 neuron** with a **sigmoid activation function** to produce a probability value between 0 and 1.
  
  - **Activation Functions:**
    - **ReLU (Rectified Linear Unit)** activation function was used for the first hidden layer to introduce non-linearity, which is effective in learning complex patterns in data.
    - **Tanh** activation function was used for the second hidden layer to capture more intricate patterns.
    - **Sigmoid** activation function was used in the output layer to predict probabilities (since it’s a binary classification problem).
  
  - **Compilation:**
    - The model was compiled using the **Adam optimizer** and **binary cross-entropy loss function**, which is standard for binary classification tasks.
    - The evaluation metric chosen is **accuracy**, which measures the overall correctness of the model.
  
  - **Training:**
    - The model was trained for **100 epochs** with the **training dataset**. Due to computational limitations, training may have been completed in a suboptimal amount of time, with the model continuing to improve beyond the set epochs.
  
  - **Evaluation Results:**
    - **Training Loss:** 0.5609
    - **Training Accuracy:** 72.75%
    - **Test Loss:** 0.5609
    - **Test Accuracy:** 72.75%
    
    These results indicate that the model achieves a reasonable classification accuracy, but there is still room for improvement.

### Summary

The deep learning model performed reasonably well, achieving approximately 72.75% accuracy on the test set. While the accuracy is acceptable, further improvements could be made by adjusting hyperparameters, increasing the dataset size, or employing feature engineering.

#### Performance comparison of the three optimized models based on accuracy and loss:

| Model | Hidden Layers & Activation Functions | Loss  | Accuracy |
|-------|--------------------------------------|-------|----------|
| **Model 1** | 2 hidden layers (Tanh, Tanh) | **0.5604** | 72.63% |
| **Model 2** | 4 hidden layers (ReLU, ReLU, Tanh, Sigmoid) | 0.5671 | **73.03%** |
| **Model 3** | 2 hidden layers (ReLU, Sigmoid) with Early Stopping | **0.5579** | 72.01% |

#### **Analysis: Which Model Performed Best?**

- **Best Accuracy:** **Model 2** performed the best in terms of accuracy (73.03%), which suggests that the additional hidden layers and mix of activation functions helped extract better patterns from the data.
- **Lowest Loss:** **Model 3** had the lowest loss (0.5579), which means it was better at minimizing prediction errors, although its accuracy was slightly lower.
- **Balanced Choice:** **Model 1** had a good trade-off between accuracy (72.63%) and loss (0.5604), performing better than Model 3 in accuracy while keeping the loss close.

---

## Alternative Model Recommendation

A potential alternative approach to solving this problem is to use a **Random Forest Classifier** or **Gradient Boosting Model (XGBoost)**. These models are well-suited for structured tabular data and often outperform deep learning models when dataset size is relatively small.

#### Table to show **best performing tree-based models** for predicting an organization's success after funding by **Alphabet Soup**:

| **Model**  | **Test Accuracy** | **Overfitting Risk** |
|------------|------------------|----------------------|
| **Random Forest**  | 74%  | Moderate |
| **XGBoost**  | 75%  | Low-Moderate |

### **Recommendation**
**For this problem, XGBoost is the best choice** because:
1. It has the **highest test accuracy (75%)**.
2. It **performs well on small-to-medium datasets**, unlike neural networks that need large datasets to generalize.
3. It **balances high recall for successful organizations** while maintaining reasonable false positives.

**If explainability is important** → **Random Forest**  
**If maximizing predictive power is key** → **XGBoost**  

---

