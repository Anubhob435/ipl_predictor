import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # Added Regressor
from xgboost import XGBClassifier, XGBRegressor # Added Regressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error # Added MAE
from sklearn.preprocessing import LabelEncoder # For encoding categorical features if needed
import joblib
import os
import numpy as np # For handling potential infinities

# Define paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(ROOT_DIR, 'ml-models')

# Create model directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load the feature-engineered data (non-encoded version first)
data_path = os.path.join(PROCESSED_DATA_DIR, 'matches_features.csv')
data = pd.read_csv(data_path)

# --- Data Preprocessing for Modeling ---
# Handle potential infinite values from economy rate calculation
data.replace([np.inf, -np.inf], np.nan, inplace=True)
# Option 1: Fill NaN economy rates (e.g., with median or a specific value)
# Calculate median per team type if possible, or global median
global_median_econ_t1 = data['team1_avg_economy_rate_last_5'].median()
global_median_econ_t2 = data['team2_avg_economy_rate_last_5'].median()
# Fix FutureWarning: Use assignment instead of inplace=True
data['team1_avg_economy_rate_last_5'] = data['team1_avg_economy_rate_last_5'].fillna(global_median_econ_t1)
data['team2_avg_economy_rate_last_5'] = data['team2_avg_economy_rate_last_5'].fillna(global_median_econ_t2)

# Option 2: Drop rows with NaN scores if they exist (already done in feature engineering)
data.dropna(subset=['winning_score', 'losing_score'], inplace=True)

# Encode categorical features (Venue, Teams, Toss Winner, Toss Decision)
# Using simple Label Encoding for now. OneHotEncoding is generally preferred for linear models/NNs
# but can work with tree-based models. Consider OneHotEncoder if performance is poor.
categorical_cols = ['venue', 'team1', 'team2', 'toss_winner', 'toss_decision']
label_encoders = {}
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le # Store encoders if needed later for prediction

# --- Save Label Encoders ---
encoder_save_path = os.path.join(MODEL_DIR, 'label_encoders.joblib')
joblib.dump(label_encoders, encoder_save_path)
print(f"Label encoders saved to {encoder_save_path}")

# Define features (X) and targets (y)
# Drop identifier columns, date, and potentially the string winner column
# CRITICAL: Also drop leaking features (scores) for the classifier task
features_to_drop_classifier = ['id', 'date', 'winner', 'inning1_score', 'inning2_score', 'winning_score', 'losing_score']
y_match_winner = data['team1_won']
y_winning_score = data['winning_score']
y_losing_score = data['losing_score']

# Ensure only existing columns are dropped for X_classifier
existing_columns_to_drop_cls = [col for col in features_to_drop_classifier if col in data.columns]
X_classifier = data.drop(columns=existing_columns_to_drop_cls + ['team1_won'])

# --- Save Expected Columns for Classifier ---
# Use X_classifier.columns AFTER it's defined
expected_columns_classifier = X_classifier.columns.tolist()
columns_save_path = os.path.join(MODEL_DIR, 'expected_columns.joblib')
try:
    joblib.dump(expected_columns_classifier, columns_save_path)
    print(f"Expected columns for classifier saved to {columns_save_path}")
except Exception as e:
    print(f"Error saving expected columns: {e}")

# For regression tasks, we might keep scores if predicting based on other features, 
# but let's use the same feature set as the classifier for consistency for now.
# If you want to predict scores using *other* scores, adjust features_to_drop_regressor accordingly.
features_to_drop_regressor = features_to_drop_classifier # Use same features for simplicity
existing_columns_to_drop_reg = [col for col in features_to_drop_regressor if col in data.columns]
X_regressor = data.drop(columns=existing_columns_to_drop_reg + ['team1_won'])

# --- Match Winner Prediction (Classification) ---
print("--- Match Winner Prediction ---")
# Split data for match winner prediction using the non-leaking feature set
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_classifier, y_match_winner, test_size=0.2, random_state=42, stratify=y_match_winner
)

# --- Hyperparameter Tuning Setup (Random Forest Classifier) ---
param_grid_rf_cls = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}
rf_cls = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search_rf_cls = GridSearchCV(estimator=rf_cls, param_grid=param_grid_rf_cls, cv=5, scoring='f1', n_jobs=-1, verbose=1) # Changed scoring to f1

# --- Run Hyperparameter Tuning (Random Forest Classifier) ---
print("Starting hyperparameter tuning for Random Forest Classifier...")
grid_search_rf_cls.fit(X_train_cls, y_train_cls)
print("Random Forest Classifier hyperparameter tuning complete.")
best_rf_cls_model = grid_search_rf_cls.best_estimator_
print(f"Best Random Forest Classifier parameters found: {grid_search_rf_cls.best_params_}")

# --- Evaluate the Best Random Forest Classifier Model ---
print("Evaluating the best Random Forest Classifier model...")
y_pred_rf_cls = best_rf_cls_model.predict(X_test_cls)
accuracy_rf_cls = accuracy_score(y_test_cls, y_pred_rf_cls)
f1_rf_cls = f1_score(y_test_cls, y_pred_rf_cls)
print(f"Best Random Forest Classifier Model Accuracy: {accuracy_rf_cls * 100:.2f}%")
print(f"Best Random Forest Classifier Model F1 Score: {f1_rf_cls:.4f}")

# --- Save the Best Random Forest Classifier Model ---
model_save_path_rf_cls = os.path.join(MODEL_DIR, 'ipl_predictor_rf_classifier_tuned.joblib')
joblib.dump(best_rf_cls_model, model_save_path_rf_cls)
print(f"Best Random Forest Classifier model saved to {model_save_path_rf_cls}")

# --- Train and Evaluate XGBoost Classifier ---
print("Training XGBoost Classifier...")
scale_pos_weight_cls = (len(y_train_cls) - sum(y_train_cls)) / sum(y_train_cls) if sum(y_train_cls) > 0 else 1
xgb_cls_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight_cls)
xgb_cls_model.fit(X_train_cls, y_train_cls)

print("Evaluating XGBoost Classifier model...")
y_pred_xgb_cls = xgb_cls_model.predict(X_test_cls)
accuracy_xgb_cls = accuracy_score(y_test_cls, y_pred_xgb_cls)
f1_xgb_cls = f1_score(y_test_cls, y_pred_xgb_cls)
print(f"XGBoost Classifier Model Accuracy: {accuracy_xgb_cls * 100:.2f}%")
print(f"XGBoost Classifier Model F1 Score: {f1_xgb_cls:.4f}")

model_save_path_xgb_cls = os.path.join(MODEL_DIR, 'ipl_predictor_xgb_classifier.joblib')
joblib.dump(xgb_cls_model, model_save_path_xgb_cls)
print(f"XGBoost Classifier model saved to {model_save_path_xgb_cls}")

# --- Feature Importance Analysis (Best Random Forest Classifier) ---
importances_cls = best_rf_cls_model.feature_importances_
# Use X_classifier.columns for feature names
feature_importance_cls_df = pd.DataFrame({'feature': X_classifier.columns, 'importance': importances_cls})
print("\nFeature Importances (Best Random Forest Classifier Model):")
print(feature_importance_cls_df.sort_values(by='importance', ascending=False).head(15))

# --- Winning Score Prediction (Regression) ---
print("\n--- Winning Score Prediction ---")
# Split data for winning score prediction using the appropriate feature set
X_train_ws, X_test_ws, y_train_ws, y_test_ws = train_test_split(
    X_regressor, y_winning_score, test_size=0.2, random_state=123 # Different random state for variety
)

# Train RandomForestRegressor
print("Training RandomForestRegressor for Winning Score...")
rf_reg_ws = RandomForestRegressor(n_estimators=100, random_state=123, n_jobs=-1, max_depth=15, min_samples_split=5)
rf_reg_ws.fit(X_train_ws, y_train_ws)

# Evaluate RandomForestRegressor
y_pred_rf_ws = rf_reg_ws.predict(X_test_ws)
mae_rf_ws = mean_absolute_error(y_test_ws, y_pred_rf_ws)
print(f"RandomForestRegressor Winning Score MAE: {mae_rf_ws:.2f}")

# Save RandomForestRegressor
model_save_path_rf_ws = os.path.join(MODEL_DIR, 'ipl_predictor_rf_winning_score.joblib')
joblib.dump(rf_reg_ws, model_save_path_rf_ws)
print(f"RandomForestRegressor Winning Score model saved to {model_save_path_rf_ws}")

# Train XGBRegressor
print("Training XGBRegressor for Winning Score...")
xgb_reg_ws = XGBRegressor(n_estimators=100, random_state=123, n_jobs=-1, learning_rate=0.1, max_depth=7)
xgb_reg_ws.fit(X_train_ws, y_train_ws)

# Evaluate XGBRegressor
y_pred_xgb_ws = xgb_reg_ws.predict(X_test_ws)
mae_xgb_ws = mean_absolute_error(y_test_ws, y_pred_xgb_ws)
print(f"XGBRegressor Winning Score MAE: {mae_xgb_ws:.2f}")

# Save XGBRegressor
model_save_path_xgb_ws = os.path.join(MODEL_DIR, 'ipl_predictor_xgb_winning_score.joblib')
joblib.dump(xgb_reg_ws, model_save_path_xgb_ws)
print(f"XGBRegressor Winning Score model saved to {model_save_path_xgb_ws}")

# --- Losing Score Prediction (Regression) ---
print("\n--- Losing Score Prediction ---")
# Split data for losing score prediction using the appropriate feature set
X_train_ls, X_test_ls, y_train_ls, y_test_ls = train_test_split(
    X_regressor, y_losing_score, test_size=0.2, random_state=456 # Different random state
)

# Train RandomForestRegressor
print("Training RandomForestRegressor for Losing Score...")
rf_reg_ls = RandomForestRegressor(n_estimators=100, random_state=456, n_jobs=-1, max_depth=15, min_samples_split=5)
rf_reg_ls.fit(X_train_ls, y_train_ls)

# Evaluate RandomForestRegressor
y_pred_rf_ls = rf_reg_ls.predict(X_test_ls)
mae_rf_ls = mean_absolute_error(y_test_ls, y_pred_rf_ls)
print(f"RandomForestRegressor Losing Score MAE: {mae_rf_ls:.2f}")

# Save RandomForestRegressor
model_save_path_rf_ls = os.path.join(MODEL_DIR, 'ipl_predictor_rf_losing_score.joblib')
joblib.dump(rf_reg_ls, model_save_path_rf_ls)
print(f"RandomForestRegressor Losing Score model saved to {model_save_path_rf_ls}")

# Train XGBRegressor
print("Training XGBRegressor for Losing Score...")
xgb_reg_ls = XGBRegressor(n_estimators=100, random_state=456, n_jobs=-1, learning_rate=0.1, max_depth=7)
xgb_reg_ls.fit(X_train_ls, y_train_ls)

# Evaluate XGBRegressor
y_pred_xgb_ls = xgb_reg_ls.predict(X_test_ls)
mae_xgb_ls = mean_absolute_error(y_test_ls, y_pred_xgb_ls)
print(f"XGBRegressor Losing Score MAE: {mae_xgb_ls:.2f}")

# Save XGBRegressor
model_save_path_xgb_ls = os.path.join(MODEL_DIR, 'ipl_predictor_xgb_losing_score.joblib')
joblib.dump(xgb_reg_ls, model_save_path_xgb_ls)
print(f"XGBRegressor Losing Score model saved to {model_save_path_xgb_ls}")

print("\nModel training script finished.")
