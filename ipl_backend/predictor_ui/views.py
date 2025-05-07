from django.shortcuts import render
import requests
import json
from .forms import PredictionForm

def index(request):
    prediction_result_dict = None # For direct template rendering
    prediction_result_json = "null" # For JavaScript
    form = PredictionForm()

    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            form_data = form.cleaned_data
            api_data = {
                "venue": form_data['venue'],
                "team1": form_data['team1'],
                "team2": form_data['team2'],
                "toss_winner": form_data['toss_winner'],
                "toss_decision": form_data['toss_decision'],
                "team1_win_rate_last_5": form_data['team1_win_rate_last_5'],
                "team1_avg_margin_last_5": form_data['team1_avg_margin_last_5'],
                "team1_avg_runs_scored_last_5": form_data['team1_avg_runs_scored_last_5'],
                "team1_avg_wickets_taken_last_5": form_data['team1_avg_wickets_taken_last_5'],
                "team1_avg_economy_rate_last_5": form_data['team1_avg_economy_rate_last_5'],
                "team2_win_rate_last_5": form_data['team2_win_rate_last_5'],
                "team2_avg_margin_last_5": form_data['team2_avg_margin_last_5'],
                "team2_avg_runs_scored_last_5": form_data['team2_avg_runs_scored_last_5'],
                "team2_avg_wickets_taken_last_5": form_data['team2_avg_wickets_taken_last_5'],
                "team2_avg_economy_rate_last_5": form_data['team2_avg_economy_rate_last_5']
            }
            
            try:
                response = requests.post(
                    'http://127.0.0.1:8000/predict',
                    json=api_data,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    prediction_data = response.json()
                    predicted_winner = form_data['team1'] if prediction_data["predicted_winner_team1"] == 1 else form_data['team2']
                    win_probability = prediction_data["prediction_probability_team1"] if prediction_data["predicted_winner_team1"] == 1 else 1 - prediction_data["prediction_probability_team1"]
                    
                    prediction_result_dict = {
                        'winner': predicted_winner,
                        'probability': round(win_probability * 100, 2),
                        'team1': form_data['team1'],
                        'team2': form_data['team2'],
                        'raw_response': prediction_data,
                        'venue': form_data['venue'] 
                    }
                else:
                    prediction_result_dict = {
                        'error': f"API Error: {response.status_code}",
                        'details': response.text
                    }
            except requests.exceptions.RequestException as e:
                prediction_result_dict = {
                    'error': "Connection Error",
                    'details': str(e),
                    'message': "Make sure the FastAPI server is running on port 8000"
                }
            
            if prediction_result_dict:
                 prediction_result_json = json.dumps(prediction_result_dict)

    return render(request, 'predictor_ui/index.html', {
        'form': form,
        'prediction_result': prediction_result_dict, # Pass the dictionary for direct display
        'prediction_result_json': prediction_result_json # Pass the JSON string for JS
    })
