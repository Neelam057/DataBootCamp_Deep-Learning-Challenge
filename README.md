# Neural Network Model for Alphabet Soup Fund Selection

## Background

The nonprofit foundation **Alphabet Soup** is seeking a tool to help identify the most promising applicants for funding. The goal is to build a machine learning model that can predict whether a given organization will be successful if funded by Alphabet Soup. So, a binary classifier will be created to assess whether applicants are successful based on various features from their data.

We have been provided with a CSV dataset containing more than **34,000** organizations that have previously received funding. The dataset includes metadata for each organization
Key columns in the dataset include:
- **EIN and NAME**: Identifiers for the organizations.
- **APPLICATION_TYPE**: The type of application made by the organization.
- **AFFILIATION**: The sector to which the organization belongs.
- **CLASSIFICATION**: The government classification of the organization.
- **USE_CASE**: The purpose for which funding is being requested.
- **ORGANIZATION**: The type of organization (nonprofit, government, etc.).
- **STATUS**: Whether the organization is active.
- **INCOME_AMT**: The income classification for the organization.
- **SPECIAL_CONSIDERATIONS**: Special factors that might affect the application.
- **ASK_AMT**: The amount of funding requested.
- **IS_SUCCESSFUL**: Target variable indicating whether the organization used the funding effectively.

## Objective
The main objective of this project is to create a **Neural Network model** that can predict the target variable **IS_SUCCESSFUL** based on the other features available in the dataset. By doing this, the model can help Alphabet Soup make data-driven decisions when selecting applicants for funding in the future.
---

## Model Design and Approach

### **1. Data Preprocessing**

1. **Data Cleaning**:
   - No null values are in this dataset so no amputation.
   - Non-numeric columns, such as **APPLICATION_TYPE**, **CLASSIFICATION** will be encoded using techniques like one-hot encoding to convert them into numerical data.
   - The **APPLICATION_TYPE** and **CLASSIFICATION** columns are cleaned and rare categories are replaced with "Other" to help reduce the number of classes in these columns.
   - The **SPECIAL_CONSIDERATIONS** column is label encoded using .map() to convert 'N' to 0 and 'Y' to 1. This converts binary data into numeric format.

2. **Feature Scaling**:
   - Since the neural network model relies on gradient-based optimization, we will scale the features to ensure that all values are within a similar range. This will help the model converge faster and prevent certain features from dominating the learning process.
   - We will use standard scaling techniques (e.g., **StandardScaler**) to standardize the feature values.
   - pd.get_dummies is used for one-hot encoding, ensuring categorical variables are properly converted to numeric form.

3. **Splitting the Data**:
   - The dataset will be split into training and testing sets. The training set will be used to train the model, while the test set will be used to evaluate the final model's performance.

---

### **2. Neural Network Architecture**

The neural network model consists of:
1. **Input Layer**: The number of input features corresponds to the number of columns after data preprocessing (scaled features).
2. **Hidden Layers**: The model contains two hidden layers:
   - The first hidden layer has 80 neurons with **ReLU activation**.
   - The second hidden layer has 30 neurons with **Tanh activation**.
3. **Output Layer**: The output layer has a single neuron with a **Sigmoid activation function** to predict the probability of success (binary classification).
4. **Loss Function**: **Binary Crossentropy** is used as the loss function for binary classification model.
5. **Optimizer**: The model uses the **Adam optimizer**, which is well-suited for training neural networks.
6. **Metrics**: **Accuracy** is used as the evaluation metric to measure the model's performance.


### **3. Model Training**
The model is trained using the training data (`X_train_scaled` and `y_train`) for 100 epochs. 

#### **4. Model Evaluation**
After training, the model's performance is evaluated on the test set (`X_test_scaled` and `y_test`), and key metrics are generated:
- **Confusion Matrix**: To show the number of true positives, false positives, true negatives, and false negatives.
- **Accuracy**: The proportion of correctly predicted samples.
- **AUC (Area Under the Curve)**: To evaluate the model's ability to discriminate between classes.

#### **5. Results**
After training the model, the following metrics were observed:
- **Train Metrics**: A confusion matrix, AUC score, and classification report for the training set.
- **Test Metrics**: A confusion matrix, AUC score, and classification report for the test set.
The results can help identify whether the model is able to distinguish between successful and unsuccessful applicants effectively.

#### **9. Conclusion and Recommendations**
- The neural network model achieved an accuracy of around 47%, but further optimization is needed.
- Some steps for model improvement could include:
  1. **Hyperparameter Tuning**: Experimenting with different numbers of layers and neurons.
  2. **Additional Preprocessing**: Trying different methods for handling imbalanced data (e.g., using class weights or oversampling the minority class).
  3. **Other Models**: A Random Forest or XGBoost model could be tested as they might perform better with structured/tabular data.

By implementing and improving this model, Alphabet Soup can make more informed decisions about which organizations are likely to succeed with funding, thereby maximizing the impact of their investments.

---
