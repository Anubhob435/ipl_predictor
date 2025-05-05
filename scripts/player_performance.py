import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV # Added GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np # Added for handling potential infinite values

print("Loading data...")
# Load data
player_stats = pd.read_csv('data/processed/player_match_stats.csv') # Corrected path
match_features = pd.read_csv('data/processed/matches_features_encoded.csv') # Corrected path
batsman_career_stats = pd.read_csv('data/processed/batsman_stats.csv') # Corrected path
bowler_career_stats = pd.read_csv('data/processed/bowler_stats.csv') # Corrected path
print("Data loaded.")

# --- Batsman Performance Prediction ---
print("Preparing data for batsman performance prediction...")

# Check columns before merging
print("Columns in player_stats:", player_stats.columns)
print("Columns in match_features:", match_features.columns)

# Merge player stats with match features
data = pd.merge(player_stats, match_features, on='match_id', how='left')
print("Columns after merging player_stats and match_features:", data.columns)

# Check columns before merging batsman career stats
print("Columns in batsman_career_stats:", batsman_career_stats.columns)
data = pd.merge(data, batsman_career_stats, on='batsman', how='left', suffixes=('', '_career'))
print("Columns after merging batsman_career_stats:", data.columns)

# Feature Engineering for Batsmen: Calculate rolling averages (e.g., last 5 matches)
data.sort_values(by=['batsman', 'date'], inplace=True) # Ensure data is sorted for rolling calculations
# Add error handling for rolling calculations in case columns don't exist as expected
try:
    data['rolling_avg_runs'] = data.groupby('batsman')['runs_scored'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    data['rolling_strike_rate'] = data.groupby('batsman')['strike_rate'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    data['rolling_balls_faced'] = data.groupby('batsman')['balls_faced'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
except KeyError as e:
    print(f"Error during batsman rolling calculation: {e}. Check column names.")
    # Decide how to handle: exit, continue without rolling features, etc.
    # For now, let's add placeholder columns if they fail
    if 'runs_scored' not in data.columns: data['rolling_avg_runs'] = np.nan
    if 'strike_rate' not in data.columns: data['rolling_strike_rate'] = np.nan
    if 'balls_faced' not in data.columns: data['rolling_balls_faced'] = np.nan

# Select relevant features for batsman model
# Using encoded features from matches_features_encoded.csv
batsman_features = [
    'venue_encoded', 'toss_decision_encoded', 'team1_encoded', 'team2_encoded',
    'team1_win_percentage_pre_match', 'team2_win_percentage_pre_match',
    'team1_points_pre_match', 'team2_points_pre_match',
    'team1_recent_form', 'team2_recent_form',
    'h2h_team1_win_percentage', 'h2h_team2_win_percentage',
    'average', 'strike_rate_career', 'total_runs', 'fifties', 'hundreds', # Career stats
    'rolling_avg_runs', 'rolling_strike_rate', 'rolling_balls_faced' # Recent form
]
target_batsman = 'runs_scored'

# Ensure all selected features actually exist in the dataframe before proceeding
existing_batsman_features = [f for f in batsman_features if f in data.columns]
missing_batsman_features = [f for f in batsman_features if f not in data.columns]
if missing_batsman_features:
    print(f"Warning: Missing batsman features: {missing_batsman_features}")
batsman_features = existing_batsman_features # Use only existing features

if target_batsman not in data.columns:
    print(f"Error: Target variable '{target_batsman}' not found in data.")
    # Handle error appropriately, maybe exit
    exit()

# Filter out rows where batsman didn't bat (runs_scored is NaN or balls_faced is 0)
# Also filter where target or features might be NaN/inf after rolling calculations
batsman_data = data.dropna(subset=[target_batsman] + batsman_features).copy()

# Check if 'balls_faced' exists before filtering
if 'balls_faced' in batsman_data.columns:
    batsman_data = batsman_data[batsman_data['balls_faced'] > 0] # Ensure batsman faced at least one ball
else:
    print("Warning: 'balls_faced' column not found for filtering batsman data.")

# Replace potential infinite values if any (though dropna should handle most)
batsman_data.replace([np.inf, -np.inf], np.nan, inplace=True)
batsman_data.dropna(subset=batsman_features, inplace=True)

print(f"Batsman data shape after cleaning: {batsman_data.shape}")

if batsman_data.empty or not batsman_features:
    print("No data or no features available for batsman model training after cleaning.")
else:
    X_batsman = batsman_data[batsman_features]
    y_batsman = batsman_data[target_batsman]

    # Split data
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_batsman, y_batsman, test_size=0.2, random_state=42)

    print("Performing hyperparameter tuning for batsman model (Random Forest Regressor)...")
    # Define the parameter grid for GridSearchCV
    param_grid_b = {
        'n_estimators': [50, 100], # Reduced for faster example, expand as needed
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3]
    }

    # Initialize the base model
    rf_b = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Initialize GridSearchCV
    grid_search_b = GridSearchCV(estimator=rf_b, param_grid=param_grid_b,
                                 cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error') # Use CV=3 for faster example

    # Fit GridSearchCV
    grid_search_b.fit(X_train_b, y_train_b)

    # Get the best estimator
    best_batsman_model = grid_search_b.best_estimator_
    print(f"Best parameters found for batsman model: {grid_search_b.best_params_}")
    print("Batsman model tuning complete.")

    # Evaluate the best model
    y_pred_b = best_batsman_model.predict(X_test_b)
    mae_b = mean_absolute_error(y_test_b, y_pred_b)
    # Calculate RMSE from the best score (which is negative MSE)
    rmse_b = np.sqrt(-grid_search_b.best_score_) # RMSE on validation sets during CV
    print(f"Batsman Model Evaluation (Best Tuned Model):")
    print(f"  Mean Absolute Error (MAE) on Test Set: {mae_b:.2f}")
    print(f"  Cross-Validated Root Mean Squared Error (RMSE): {rmse_b:.2f}") # Note: This is from CV, not test set

    # Save the best model
    model_path_b = 'ml-models/batsman_performance_rf_tuned_model.joblib' # Changed filename
    joblib.dump(best_batsman_model, model_path_b)
    print(f"Tuned batsman performance model saved to {model_path_b}")

# --- Bowler Performance Prediction (Placeholder) ---
print("\nPreparing data for bowler performance prediction...")

# Check columns before merging bowler career stats
print("Columns in bowler_career_stats:", bowler_career_stats.columns)
# Ensure 'bowler' column exists in 'data' before merging
if 'bowler' not in data.columns:
    print("Error: 'bowler' column not found in data before merging bowler stats.")
    # Handle error, maybe skip bowler part or exit
    exit()

data = pd.merge(data, bowler_career_stats, on='bowler', how='left', suffixes=('', '_career_bowl'))
print("Columns after merging bowler_career_stats:", data.columns)

# Feature Engineering for Bowlers: Calculate rolling averages (e.g., last 5 matches)
# Ensure data is sorted by bowler and date
data.sort_values(by=['bowler', 'date'], inplace=True)
# Add error handling
try:
    data['rolling_avg_wickets'] = data.groupby('bowler')['wickets_taken'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    data['rolling_economy'] = data.groupby('bowler')['economy_rate'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    data['rolling_runs_conceded'] = data.groupby('bowler')['runs_conceded'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
except KeyError as e:
    print(f"Error during bowler rolling calculation: {e}. Check column names.")
    # Add placeholder columns if they fail
    if 'wickets_taken' not in data.columns: data['rolling_avg_wickets'] = np.nan
    if 'economy_rate' not in data.columns: data['rolling_economy'] = np.nan
    if 'runs_conceded' not in data.columns: data['rolling_runs_conceded'] = np.nan

# Select relevant features for bowler model
bowler_features = [
    'venue_encoded', 'toss_decision_encoded', 'team1_encoded', 'team2_encoded',
    'team1_win_percentage_pre_match', 'team2_win_percentage_pre_match',
    'team1_points_pre_match', 'team2_points_pre_match',
    'team1_recent_form', 'team2_recent_form',
    'h2h_team1_win_percentage', 'h2h_team2_win_percentage',
    'wickets', 'economy', 'runs_conceded_career_bowl', 'overs_bowled_career_bowl',# Career stats (using unique suffixes from merge)
    'rolling_avg_wickets', 'rolling_economy', 'rolling_runs_conceded', 'rolling_overs_bowled' # Recent form
]
target_bowler = 'wickets_taken'

# Ensure all selected features actually exist in the dataframe before proceeding
existing_bowler_features = [f for f in bowler_features if f in data.columns]
missing_bowler_features = [f for f in bowler_features if f not in data.columns]
if missing_bowler_features:
    print(f"Warning: Missing bowler features: {missing_bowler_features}")
bowler_features = existing_bowler_features # Use only existing features

if target_bowler not in data.columns:
    print(f"Error: Target variable '{target_bowler}' not found in data.")
    # Handle error appropriately
    exit()

# Filter out rows where bowler didn't bowl (wickets_taken is NaN or overs_bowled is 0)
# Also filter where target or features might be NaN/inf
bowler_data_final = data.dropna(subset=[target_bowler] + bowler_features).copy()

# Check if 'overs_bowled' exists before filtering
if 'overs_bowled' in bowler_data_final.columns:
    bowler_data_final = bowler_data_final[bowler_data_final['overs_bowled'] > 0] # Ensure bowler bowled at least one ball
else:
    print("Warning: 'overs_bowled' column not found for filtering bowler data.")

# Replace potential infinite values if any
bowler_data_final.replace([np.inf, -np.inf], np.nan, inplace=True)
bowler_data_final.dropna(subset=bowler_features, inplace=True)

print(f"Bowler data shape after cleaning: {bowler_data_final.shape}")

if bowler_data_final.empty or not bowler_features:
    print("No data or no features available for bowler model training after cleaning.")
else:
    X_bowler = bowler_data_final[bowler_features]
    y_bowler = bowler_data_final[target_bowler]

    # Split data
    X_train_bw, X_test_bw, y_train_bw, y_test_bw = train_test_split(X_bowler, y_bowler, test_size=0.2, random_state=42)

    print("Performing hyperparameter tuning for bowler model (Random Forest Regressor)...")
    # Define the parameter grid for GridSearchCV
    param_grid_bw = {
        'n_estimators': [50, 100], # Reduced for faster example
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3]
    }

    # Initialize the base model
    rf_bw = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Initialize GridSearchCV
    grid_search_bw = GridSearchCV(estimator=rf_bw, param_grid=param_grid_bw,
                                  cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error') # Use CV=3 for faster example

    # Fit GridSearchCV
    grid_search_bw.fit(X_train_bw, y_train_bw)

    # Get the best estimator
    best_bowler_model = grid_search_bw.best_estimator_
    print(f"Best parameters found for bowler model: {grid_search_bw.best_params_}")
    print("Bowler model tuning complete.")

    # Evaluate the best model
    y_pred_bw = best_bowler_model.predict(X_test_bw)
    mae_bw = mean_absolute_error(y_test_bw, y_pred_bw)
    # Calculate RMSE from the best score (negative MSE)
    rmse_bw = np.sqrt(-grid_search_bw.best_score_) # RMSE on validation sets during CV
    print(f"Bowler Model Evaluation (Best Tuned Model):")
    print(f"  Mean Absolute Error (MAE) on Test Set: {mae_bw:.2f}")
    print(f"  Cross-Validated Root Mean Squared Error (RMSE): {rmse_bw:.2f}") # Note: This is from CV, not test set

    # Save the best model
    model_path_bw = 'ml-models/bowler_performance_rf_tuned_model.joblib' # Changed filename
    joblib.dump(best_bowler_model, model_path_bw)
    print(f"Tuned bowler performance model saved to {model_path_bw}")

print("\nPlayer performance script finished.")
