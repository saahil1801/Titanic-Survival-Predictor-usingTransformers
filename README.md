# Titanic-Survival-Predictor-usingTransformers

Titanic Survival Predictor: A data analysis and machine learning project that leverages powerful transformers to predict passenger survival on the Titanic. Explore the fascinating history of the Titanic and harness the capabilities of state-of-the-art data preprocessing techniques. Discover insights into passenger demographics while building and evaluating predictive models with the help of these transformers. 

## Dataset Overview

This project utilizes the Titanic dataset, which contains passenger information from the historic Titanic voyage. The dataset includes details such as passenger names, ages, genders, ticket classes, fares, cabin information, and whether passengers survived or not. The primary goal is to predict passenger survival based on various features, making it a classic classification problem.

## Data Transformation with Transformers

To prepare the data for modeling, we employ a series of data transformation steps using scikit-learn transformers:

- **Label Encoding**: Categorical features like 'Sex' and 'Embarked' are label-encoded to convert them into numerical form.

- **Missing Data Imputation**: We use the `SimpleImputer` to fill missing values in features such as 'Age' and 'Embarked' with appropriate values.

- **Column Transformation**: We use the `ColumnTransformer` to apply specific transformations to numerical and categorical features separately.

- **Standard Scaling**: Numerical features like 'Age' and 'Fare' are scaled using `StandardScaler` to bring them to a common scale.

- **One-Hot Encoding**: Categorical features are further processed using one-hot encoding (`OneHotEncoder`) to convert them into a binary format.

### Feature Selection

- **Focused Feature Selection**: We carefully selected relevant features for prediction while excluding less influential ones. This reduced the dimensionality of the dataset and improved model efficiency.

To address class imbalance, we utilize the `RandomOverSampler` from the `imblearn` library to balance the distribution of the target variable.

### Model Ensemble and Resampling

To build a robust predictive model, we employ a diverse set of classification algorithms, including Random Forest, Gradient Boosting, AdaBoost, Bagging, Support Vector Machine (SVM), Decision Tree, and XGBoost. These models are combined into a powerful ensemble using the `VotingClassifier`.

## Efficient Optimization for Runtime and Accuracy

One of the key achievements of this project is the optimization of code to achieve impressive results within a short runtime. Here's how we optimized the code for efficiency while maintaining a high accuracy rate of 80%:

This project aims to showcase how a combination of transformers and ensemble methods can be applied to real-world datasets for accurate predictions and insights.
