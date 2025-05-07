from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Load environment variables (create a .env file with your API key)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

def analyze_match_prediction(match_data, prediction_result):
    """
    Analyze match data and prediction result using Gemini AI.
    
    Args:
        match_data (dict): Match details and team statistics
        prediction_result (dict): Prediction results from the model
        
    Returns:
        str: AI analysis of the match prediction
    """
    # Create a prompt with all relevant match information
    prompt = f"""
    Analyze this IPL cricket match prediction and explain the reasoning:
    
    Match Details:
    - Venue: {match_data.get('venue')}
    - Team 1: {match_data.get('team1')}
    - Team 2: {match_data.get('team2')}
    - Toss Winner: {match_data.get('toss_winner')}
    - Toss Decision: {match_data.get('toss_decision')}
    
    Team Statistics (Last 5 Matches):
    
    {match_data.get('team1')}:
    - Win Rate: {match_data.get('team1_win_rate_last_5')}
    - Average Margin: {match_data.get('team1_avg_margin_last_5')}
    - Average Runs Scored: {match_data.get('team1_avg_runs_scored_last_5')}
    - Average Wickets Taken: {match_data.get('team1_avg_wickets_taken_last_5')}
    - Average Economy Rate: {match_data.get('team1_avg_economy_rate_last_5')}
    
    {match_data.get('team2')}:
    - Win Rate: {match_data.get('team2_win_rate_last_5')}
    - Average Margin: {match_data.get('team2_avg_margin_last_5')}
    - Average Runs Scored: {match_data.get('team2_avg_runs_scored_last_5')}
    - Average Wickets Taken: {match_data.get('team2_avg_wickets_taken_last_5')}
    - Average Economy Rate: {match_data.get('team2_avg_economy_rate_last_5')}
    
    Prediction Result:
    - Predicted Winner: {prediction_result.get('winner')}
    - Win Probability: {prediction_result.get('probability')}%
    
    Please provide an insightful analysis that includes:
    1. Key factors that influenced this prediction
    2. Team strengths and weaknesses based on the statistics
    3. How the toss outcome might influence the match
    4. Any historical context or patterns that might be relevant
    5. Potential scenarios for how the match might play out
    
    Format your response in clear paragraphs with proper headings.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=800,
                temperature=0.3
            )
        )
        
        return response.text
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_match = {
        "venue": "Eden Gardens",
        "team1": "Kolkata Knight Riders",
        "team2": "Mumbai Indians",
        "toss_winner": "Mumbai Indians",
        "toss_decision": "bat",
        "team1_win_rate_last_5": 0.6,
        "team1_avg_margin_last_5": 15.2,
        "team1_avg_runs_scored_last_5": 178.4,
        "team1_avg_wickets_taken_last_5": 7.2,
        "team1_avg_economy_rate_last_5": 8.3,
        "team2_win_rate_last_5": 0.4,
        "team2_avg_margin_last_5": -5.8,
        "team2_avg_runs_scored_last_5": 165.6,
        "team2_avg_wickets_taken_last_5": 6.8,
        "team2_avg_economy_rate_last_5": 8.7
    }
    
    sample_prediction = {
        "winner": "Kolkata Knight Riders",
        "probability": 65.75
    }
    
    analysis = analyze_match_prediction(sample_match, sample_prediction)
    print(analysis)