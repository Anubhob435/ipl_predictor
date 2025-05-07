import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException # Import HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
import uvicorn
import numpy as np # Import numpy

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, 'ml-models')
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, 'ipl_predictor_rf_classifier_tuned.joblib') # Or choose xgb
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoders.joblib')
# Define path to save/load expected columns
COLUMNS_PATH = os.path.join(MODEL_DIR, 'expected_columns.joblib')

# --- Load Model, Encoders, and Columns ---
try:
    model = joblib.load(CLASSIFIER_MODEL_PATH)
    label_encoders = joblib.load(ENCODER_PATH)
    # Load expected columns from file
    expected_columns = joblib.load(COLUMNS_PATH)
    print("Model, encoders, and expected columns loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: File not found. {e}. Ensure model, encoders, and columns file exist.")
    # Attempt to load columns from training script as a fallback (requires running training first)
    try:
        # This part assumes you have a way to get columns from training script output or a saved file
        # If model_training.py saves columns, load them here. Otherwise, hardcode as last resort.
        print("Attempting fallback column loading (requires recent training run)...")
        # Example: Re-run relevant part of training script logic if needed (less ideal)
        # Or load from a CSV/text file if saved separately
        # Hardcoding as a temporary measure if fallback fails:
        expected_columns = [
            'venue', 'team1', 'team2', 'toss_winner', 'toss_decision',
            'team1_win_rate_last_5', 'team1_avg_margin_last_5',
            'team2_win_rate_last_5', 'team2_avg_margin_last_5',
            'team1_avg_runs_scored_last_5', 'team1_avg_wickets_taken_last_5', 'team1_avg_economy_rate_last_5',
            'team2_avg_runs_scored_last_5', 'team2_avg_wickets_taken_last_5', 'team2_avg_economy_rate_last_5',
            'toss_winner_is_team1', 'team1_bat_first', 'team1_toss_bat_first', 'team2_toss_bat_first'
        ]
        print("Warning: Using hardcoded expected columns. Run training script to generate columns file for robustness.")
    except Exception as fallback_e:
        print(f"Fallback column loading failed: {fallback_e}")
        model = None
        label_encoders = None
        expected_columns = []

except Exception as e:
    print(f"An error occurred during loading: {e}")
    model = None
    label_encoders = None
    expected_columns = []


# --- API Definition ---
app = FastAPI(title="IPL Match Winner Predictor API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],  # Django server URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Import the Gemini analysis function
import sys
import os

# Make sure llm_vizualization is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_gemini_import_error_details = "" # To store specific error details
try:
    from llm_vizualization.model_gemini import analyze_match_prediction
    print("Successfully imported analyze_match_prediction from llm_vizualization.model_gemini")
except Exception as e:
    #_gemini_import_error_details = f"{type(e).__name__}: {str(e)}"
    print(f"Error importing 'analyze_match_prediction' from 'llm_vizualization.model_gemini':")
    # Define a dummy function as fallback, capturing the error detail
    #def analyze_match_prediction(match_data, prediction_result):
        #return f"Error: Gemini analysis function could not be loaded. Import failed with: {_gemini_import_error_details}. Please check your setup, ensure GEMINI_API_KEY is in .env, and all dependencies are installed."

# Define input data model using Pydantic
# Match the features used in X_classifier during training
class MatchInput(BaseModel):
    venue: str
    team1: str
    team2: str
    toss_winner: str
    toss_decision: str # 'bat' or 'field'

    team1_win_rate_last_5: float = 0.5 # Example default
    team1_avg_margin_last_5: float = 0.0
    team2_win_rate_last_5: float = 0.5
    team2_avg_margin_last_5: float = 0.0
    team1_avg_runs_scored_last_5: float = 150.0
    team1_avg_wickets_taken_last_5: float = 5.0
    team1_avg_economy_rate_last_5: float = 8.0
    team2_avg_runs_scored_last_5: float = 150.0
    team2_avg_wickets_taken_last_5: float = 5.0
    team2_avg_economy_rate_last_5: float = 8.0

# Define output data model
class PredictionOutput(BaseModel):
    predicted_winner_team1: int # 1 if team1 predicted to win, 0 otherwise
    prediction_probability_team1: float # Probability of team1 winning

# Request model for AI Analysis
class AIAnalysisRequest(BaseModel):
    match_data: dict
    prediction_result: dict

# Response model for AI Analysis
class AIAnalysisResponse(BaseModel):
    analysis: str

@app.post("/predict", response_model=PredictionOutput)
async def predict_winner(match_input: MatchInput):
    if not model or not label_encoders or not expected_columns:
         raise HTTPException(status_code=503, detail="Model, encoders, or expected columns not loaded. Cannot predict.")

    # 1. Create DataFrame from input
    input_data = pd.DataFrame([match_input.dict()])
    raw_toss_decision = match_input.toss_decision # Store raw value before encoding

    # 2. Preprocess: Apply Label Encoding
    try:
        for col, le in label_encoders.items():
            if col in input_data.columns:
                # Handle unseen labels during prediction
                current_value = input_data[col].iloc[0]
                if current_value in le.classes_:
                    input_data[col] = le.transform([current_value])[0]
                else:
                    # Handle unseen label: Option 1: Raise Error
                    raise HTTPException(status_code=400, detail=f"Unseen value '{current_value}' for feature '{col}'. Check input or retrain model/encoders.")
                    # Option 2: Use a default value (e.g., -1 or median encoding) - Requires model to handle it
                    # input_data[col] = -1
                    # print(f"Warning: Unseen label '{current_value}' encountered in column '{col}'. Using default value.")

    except HTTPException as e:
        raise e # Re-raise the HTTP exception
    except Exception as e:
        print(f"Error during encoding: {e}")
        raise HTTPException(status_code=500, detail=f"Error during data encoding: {e}")


    # 3. Add derived features (toss advantage) - Ensure this matches training!
    # Compare encoded values for toss_winner vs team1
    input_data['toss_winner_is_team1'] = (input_data['toss_winner'] == input_data['team1']).astype(int)
    # Use the raw string value for toss_decision comparison
    # Fix: Convert boolean directly to int
    input_data['team1_bat_first'] = int(raw_toss_decision == 'bat')
    input_data['team1_toss_bat_first'] = input_data['toss_winner_is_team1'] * input_data['team1_bat_first']
    input_data['team2_toss_bat_first'] = (1 - input_data['toss_winner_is_team1']) * input_data['team1_bat_first']


    # 4. Ensure column order and presence matches training data
    try:
        # Add missing columns with default value (e.g., 0 or NaN) if applicable, though ideally input should be complete
        for col in expected_columns:
             if col not in input_data.columns:
                 print(f"Warning: Input missing expected column '{col}'. Adding with default value 0.")
                 input_data[col] = 0 # Or np.nan, depending on model handling

        # Reorder columns
        input_data = input_data[expected_columns]
    except KeyError as e:
        # This error shouldn't happen if missing columns are handled above, but keep as safeguard
        print(f"Column mismatch error during reordering: {e}")
        raise HTTPException(status_code=400, detail=f"Input data column mismatch error: {e}")
    except Exception as e:
        print(f"Error ensuring column order: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing input columns: {e}")


    # 5. Make Prediction
    try:
        # Ensure data types are correct (e.g., float for numerical features)
        input_data = input_data.astype(float) # Or specify dtypes more granularly if needed
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        # Print input_data dtypes for debugging
        print("Input data dtypes:\n", input_data.dtypes)
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {e}")


    # Assuming class 1 corresponds to team1 winning
    # Ensure probabilities array has expected length
    if len(probabilities) > 1:
        prob_team1_wins = probabilities[1]
    else:
        # Handle case where predict_proba might only return one value (e.g., if only one class possible)
        prob_team1_wins = probabilities[0] if prediction == 1 else 1 - probabilities[0]
        print("Warning: predict_proba returned only one probability value.")


    return PredictionOutput(
        predicted_winner_team1=int(prediction),
        prediction_probability_team1=float(prob_team1_wins)
    )

@app.post("/analyze", response_model=AIAnalysisResponse)
async def analyze_match(analysis_request: AIAnalysisRequest):
    """
    Analyze match prediction using Gemini AI
    """
    try:
        analysis = analyze_match_prediction(
            analysis_request.match_data, 
            analysis_request.prediction_result
        )
        
        return AIAnalysisResponse(analysis=analysis)
    except Exception as e:
        print(f"Error during AI analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")

# --- Function to save expected columns (Run this from model_training.py) ---
# def save_expected_columns(columns, path):
#     try:
#         joblib.dump(columns, path)
#         print(f"Expected columns saved to {path}")
#     except Exception as e:
#         print(f"Error saving expected columns: {e}")

# --- Run the API ---
if __name__ == "__main__":
    if model and label_encoders and expected_columns:
        print("Starting FastAPI server...")
        # Use host="0.0.0.0" to make it accessible on your network
        uvicorn.run("fast_api:app", host="127.0.0.1", port=8000, reload=True) # Use reload for development
    else:
        print("Could not start server because model, encoders, or expected columns failed to load.")
