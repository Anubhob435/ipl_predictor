# scripts/hyper_tuning.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import numpy as np # For handling potential inf values if needed
import os

print("Starting hyperparameter tuning script...")
start_time = time.time()

# --- Configuration ---
DATA_PATH = 'data/processed/matches_features_encoded.csv'
MODEL_DIR = 'ml-models'
TARGET_VARIABLE = 'winner_encoded' # Assuming 0 or 1 indicating team1 or team2 win
FEATURES = [ # Adjust this list based on your actual feature engineering
    'venue_encoded', 'toss_decision_encoded', 'team1_encoded', 'team2_encoded',
    'team1_win_percentage_pre_match', 'team2_win_percentage_pre_match',
    'team1_points_pre_match', 'team2_points_pre_match',
    'team1_recent_form', 'team2_recent_form',
    'h2h_team1_win_percentage', 'h2h_team2_win_percentage'
]
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3 # Use 3 for faster execution, increase (e.g., 5 or 10) for more robust tuning

# --- Create Model Directory if it doesn't exist ---
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load Data ---
print(f"Loading processed match data from {DATA_PATH}...")
try:
    data = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")
    print(f"Data shape: {data.shape}")
    # print("Columns:", data.columns.tolist()) # Uncomment to verify columns
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Feature Selection and Data Prep ---
print("Preparing data for tuning...")

# Verify features and target exist
missing_features = [f for f in FEATURES if f not in data.columns]
if missing_features:
    print(f"Error: The following features are missing from the data: {missing_features}")
    exit()
if TARGET_VARIABLE not in data.columns:
    print(f"Error: Target column '{TARGET_VARIABLE}' not found in the data.")
    exit()

# Handle potential NaN/inf values before splitting
data.replace([np.inf, -np.inf], np.nan, inplace=True)
# Consider more sophisticated imputation if needed, e.g., data.fillna(data.median(), inplace=True)
data.dropna(subset=FEATURES + [TARGET_VARIABLE], inplace=True) # Drop rows with NaNs in features or target

if data.empty:
    print("Error: No data remaining after cleaning NaN/inf values.")
    exit()

X = data[FEATURES]
y = data[TARGET_VARIABLE]

print(f"Features selected ({len(FEATURES)}): {FEATURES}")
print(f"Target variable: {TARGET_VARIABLE}")
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
print(f"Target value counts:\n{y.value_counts(normalize=True)}") # Check class balance

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y # Important for classification tasks, especially if imbalanced
)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# --- Random Forest Hyperparameter Tuning ---
print(f"\n--- Tuning RandomForestClassifier (CV={CV_FOLDS}) ---")
# Define a smaller grid for faster demonstration
rf_param_grid = {
    'n_estimators': [100, 200], # Number of trees
    'max_depth': [10, 20, None], # Max depth of trees
    'min_samples_split': [2, 5], # Min samples to split a node
    'min_samples_leaf': [1, 3], # Min samples in a leaf node
    'class_weight': ['balanced', None] # Handle class imbalance
}

rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid,
                              cv=CV_FOLDS, n_jobs=-1, verbose=1, scoring='accuracy')

print("Fitting GridSearchCV for RandomForestClassifier...")
rf_start_time = time.time()
rf_grid_search.fit(X_train, y_train)
rf_end_time = time.time()
print(f"RandomForest tuning took {rf_end_time - rf_start_time:.2f} seconds.")

best_rf_model = rf_grid_search.best_estimator_
print(f"Best parameters found for RandomForestClassifier: {rf_grid_search.best_params_}")
print(f"Best cross-validation accuracy for RandomForestClassifier: {rf_grid_search.best_score_:.4f}")

# Evaluate best RF model on test set
y_pred_rf = best_rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nBest RandomForestClassifier Test Set Accuracy: {accuracy_rf:.4f}")
print("Best RandomForestClassifier Test Set Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Save the tuned RF model
rf_model_path = os.path.join(MODEL_DIR, 'ipl_predictor_rf_tuned_model.joblib')
joblib.dump(best_rf_model, rf_model_path)
print(f"Tuned RandomForestClassifier model saved to {rf_model_path}")

# --- XGBoost Hyperparameter Tuning ---
print(f"\n--- Tuning XGBClassifier (CV={CV_FOLDS}) ---")
# Define a smaller grid for faster demonstration
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1], # Usually 0.01 to 0.2
    'subsample': [0.8, 1.0], # Fraction of samples used per tree
    'colsample_bytree': [0.8, 1.0], # Fraction of features used per tree
    # 'gamma': [0, 0.1] # Minimum loss reduction required to make a further partition
}

# Use scale_pos_weight for imbalanced classes if needed
# Calculate it: scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
xgb = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss', n_jobs=-1) # Add scale_pos_weight=scale_pos_weight if needed

xgb_grid_search = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid,
                               cv=CV_FOLDS, n_jobs=-1, verbose=1, scoring='accuracy')

print("Fitting GridSearchCV for XGBClassifier...")
xgb_start_time = time.time()
xgb_grid_search.fit(X_train, y_train)
xgb_end_time = time.time()
print(f"XGBoost tuning took {xgb_end_time - xgb_start_time:.2f} seconds.")


best_xgb_model = xgb_grid_search.best_estimator_
print(f"Best parameters found for XGBClassifier: {xgb_grid_search.best_params_}")
print(f"Best cross-validation accuracy for XGBClassifier: {xgb_grid_search.best_score_:.4f}")

# Evaluate best XGB model on test set
y_pred_xgb = best_xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"\nBest XGBClassifier Test Set Accuracy: {accuracy_xgb:.4f}")
print("Best XGBClassifier Test Set Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Save the tuned XGB model
xgb_model_path = os.path.join(MODEL_DIR, 'ipl_predictor_xgb_tuned_model.joblib')
joblib.dump(best_xgb_model, xgb_model_path)
print(f"Tuned XGBClassifier model saved to {xgb_model_path}")

# --- Script Completion ---
end_time = time.time()
print(f"\nHyperparameter tuning script finished in {end_time - start_time:.2f} seconds.")