import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load Data
# Ensure 'hr_data.csv' is in the same folder
print("Loading data...")
file_name = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/hr_data.csv"
df = pd.read_csv(file_name)


# 2. Define Features and Target
# Select specific features available in the dashboard
features = [
    'number_project', 'average_montly_hours', 'time_spend_company', 
    'Work_accident', 'promotion_last_5years', 'department', 'salary'
]
X = df[features]
y = df['left']

# 3. Define Preprocessing Pipeline
# Numerical features (to be scaled)
numeric_features = ['number_project', 'average_montly_hours', 'time_spend_company']

# Categorical features (to be One-Hot encoded)
categorical_features = ['department']

# Ordinal features (to be ordered: low < medium < high)
ordinal_features = ['salary']
salary_order = [['low', 'medium', 'high']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('ord', OrdinalEncoder(categories=salary_order), ordinal_features)
    ],
    remainder='passthrough' # Keep binary columns like Work_accident
)

# 4. Create the Model Pipeline
# Using Class Weight='balanced' to handle the fact that fewer people leave than stay
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# 5. Train the Model
print("Training model...")
# Optional: Split to check accuracy during this run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf.fit(X_train, y_train)

# Evaluate
print("Model Performance on Test Set:")
print(classification_report(y_test, clf.predict(X_test)))
print(f"ROC AUC Score: {roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]):.2f}")

# 6. Finalize and Save
# Retrain on full dataset for production use
print("Retraining on full dataset for production...")
clf.fit(X, y)

# Save the pipeline to a file
joblib.dump(clf, 'final_model.joblib')
print("[OK] Model saved as 'final_model.joblib'")