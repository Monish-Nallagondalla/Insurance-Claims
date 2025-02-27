# Insurance Claims Fraud Detection with Confusion Matrix

This project aims to detect fraudulent insurance claims using a Logistic Regression model and evaluate its performance with a confusion matrix. The dataset contains various features related to insurance policies, customer details, and claims, with the target variable indicating whether a claim was fraudulent.

## Project Overview

- **Objective**: Predict whether an insurance claim is fraudulent (`Y` or `N`) using machine learning.
- **Dataset**: `insurance_claims.csv` - Contains 40 columns with details about insurance policies and claims.
- **Model**: Logistic Regression from scikit-learn.
- **Evaluation**: Confusion matrix, precision, recall, F1-score, and accuracy.
- **Visualizations**: Bar plots for fraud counts and incident states.

## Dataset

The dataset (`insurance_claims.csv`) includes 1000 rows and 40 columns. Key columns include:
- `months_as_customer`: Number of months as a customer.
- `age`: Age of the insured.
- `policy_state`: State where the policy was issued.
- `total_claim_amount`: Total amount claimed.
- `fraud_reported`: Target variable (`Y` for fraud, `N` for no fraud).

Sample data:

| months_as_customer | age | policy_state | total_claim_amount | fraud_reported |
|--------------------|-----|--------------|--------------------|----------------|
| 328                | 48  | OH           | 71610             | Y              |
| 228                | 42  | IN           | 5070              | Y              |
| 134                | 29  | OH           | 34650             | N              |
| 256                | 41  | IL           | 63400             | Y              |
| 228                | 44  | IL           | 6500              | N              |

## Prerequisites

To run this project, you need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Install the dependencies using pip:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Prepare the Dataset**:
   - Place `insurance_claims.csv` in the project directory.

3. **Run the Code**:
   - Open the `.ipynb` file in Jupyter Notebook:
     ```bash
     jupyter notebook Insurance_Claims_Fraud_Detection.ipynb
     ```
   - Alternatively, convert it to a `.py` file and run:
     ```bash
     python insurance_claims_fraud_detection.py
     ```

4. **Output**:
   - Visualizations: Bar plots showing fraud counts and incident state distribution.
   - Model Performance: Confusion matrix and classification report printed to the console.

## Code Breakdown

- **Data Loading**: Loads the CSV file into a pandas DataFrame.
- **Preprocessing**:
  - Drops irrelevant columns (`policy_number`, `policy_bind_date`, `insured_zip`, `_c39`).
  - Encodes categorical variables using `pd.get_dummies`.
  - Converts target variable (`fraud_reported`) to binary (1 for `Y`, 0 for `N`).
- **Model Training**: Splits data into 80% training and 20% testing sets, then fits a Logistic Regression model.
- **Evaluation**: Generates predictions and evaluates them with a confusion matrix and classification report.

## Results

### Fraud Distribution
- Non-fraudulent (`N`): 753
- Fraudulent (`Y`): 247

### Model Performance
- **Confusion Matrix**:
  ```
  [[153   1]
   [ 46   0]]
  ```
  - True Negatives (TN): 153
  - False Positives (FP): 1
  - False Negatives (FN): 46
  - True Positives (TP): 0

- **Classification Report**:
  ```
              precision    recall  f1-score   support
         0       0.77      0.99      0.87       154
         1       0.00      0.00      0.00        46
  accuracy                            0.77       200
  macro avg       0.38      0.50      0.43       200
  weighted avg    0.59      0.77      0.67       200
  ```

- **Observations**:
  - The model predicts non-fraudulent claims well (high recall for class 0), but fails to detect fraudulent claims (recall of 0 for class 1).
  - Accuracy is 77%, but the model is biased toward the majority class (non-fraud).

## Visualizations

1. **Fraud Count Bar Plot**: Displays the count of fraudulent vs. non-fraudulent claims.
2. **Incident State Bar Plot**: Shows the distribution of claims across states (e.g., NY, SC, WV).

## Future Improvements

- Address the convergence warning by:
  - Increasing `max_iter` in Logistic Regression.
  - Scaling the data using `StandardScaler`.
- Experiment with other models (e.g., Random Forest, Decision Tree) for better fraud detection.
- Handle class imbalance using techniques like SMOTE or oversampling.
- Tune hyperparameters to improve recall for the fraud class.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python, scikit-learn, pandas, and matplotlib.
```

### Instructions
1. Copy the entire code block above.
2. Open or create a `README.md` file in your GitHub repository.
3. Paste the code into the file.
4. Replace `your-username` and `your-repo-name` in the "Clone the Repository" section with your actual GitHub username and repository name.
5. Save and commit the file to your repository.
