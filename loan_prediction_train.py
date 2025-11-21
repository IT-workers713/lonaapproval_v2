# loan_prediction_train.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load dataset (replace with your Kaggle file path)
df = pd.read_csv("loan_train.csv")  # Kaggle dataset

# 2. Drop Loan_ID (not useful for prediction)
df.drop("Loan_ID", axis=1, inplace=True)

# 3. Encode target variable
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# 4. Separate features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# 5. Identify categorical and numeric columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# 6. Preprocessing pipelines
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(transformers=[
    ('categorical', cat_pipeline, cat_cols),
    ('numerical', num_pipeline, num_cols)
])

# 7. Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 9. Train model
model.fit(X_train, y_train)

# 10. Predictions
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 11. Feature importance extraction
rf_model = model.named_steps['clf']
ohe = model.named_steps['preprocessor'].named_transformers_['categorical'].named_steps['encoder']
encoded_cat_features = ohe.get_feature_names_out(cat_cols)
all_feature_names = np.concatenate([encoded_cat_features, num_cols])

importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 12. Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(data=feat_imp_df.head(10), x="Importance", y="Feature", palette="viridis")
plt.title("Top 10 Features Influencing Loan Approval")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# 13. Save model
joblib.dump(model, "loan_pipeline.joblib")
print("Model saved as loan_pipeline.joblib")
