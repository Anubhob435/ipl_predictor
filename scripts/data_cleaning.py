"""
Feature Engineering Script for IPL Match Outcome Prediction
This script processes raw IPL match data and creates engineered features for model training
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Define paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

# Create processed directory if it doesn't exist
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

def load_data():
    """Load the raw IPL data"""
    matches_path = os.path.join(RAW_DATA_DIR, 'matches.csv')
    deliveries_path = os.path.join(RAW_DATA_DIR, 'deliveries.csv')
    
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    
    print(f"Loaded {len(matches)} matches and {len(deliveries)} delivery records")
    return matches, deliveries

def preprocess_matches(matches):
    """Preprocess the matches data"""
    matches_processed = matches.copy()
    
    # Convert date to datetime
    matches_processed['date'] = pd.to_datetime(matches_processed['date'])
    
    # Extract year and create a season feature if not already present
    if 'season' not in matches_processed.columns:
        matches_processed['season'] = matches_processed['date'].dt.year
    
    # Remove matches with no result
    matches_processed = matches_processed[matches_processed['result'] != 'no result']
    
    # Ensure team names are consistent (normalize team names)
    team_name_mapping = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Deccan Chargers': 'Sunrisers Hyderabad',
        'Rising Pune Supergiants': 'Rising Pune Supergiant',
        'Kings XI Punjab': 'Punjab Kings'
        # Add more mappings if needed
    }
    
    for old_name, new_name in team_name_mapping.items():
        matches_processed['team1'] = matches_processed['team1'].replace(old_name, new_name)
        matches_processed['team2'] = matches_processed['team2'].replace(old_name, new_name)
        matches_processed['winner'] = matches_processed['winner'].replace(old_name, new_name)
        matches_processed['toss_winner'] = matches_processed['toss_winner'].replace(old_name, new_name)
    
    return matches_processed

def create_team_features(matches, deliveries):
    """Create team-based features"""
    # Compute team performance metrics
    team_features = {}
    
    # Process matches data to get all unique teams
    all_teams = pd.unique(matches[['team1', 'team2']].values.ravel('K'))
    
    for team in all_teams:
        if pd.isna(team):
            continue
        
        # Matches won by the team
        team_wins = matches[matches['winner'] == team]
        
        # Matches played by the team
        team_matches = matches[(matches['team1'] == team) | (matches['team2'] == team)]
        
        # Win rate
        win_rate = len(team_wins) / len(team_matches) if len(team_matches) > 0 else 0
        
        # Toss win rate
        toss_wins = matches[matches['toss_winner'] == team]
        toss_win_rate = len(toss_wins) / len(team_matches) if len(team_matches) > 0 else 0
        
        # Home vs Away performance
        home_matches = matches[matches['team1'] == team]  # Assuming team1 is home team
        away_matches = matches[matches['team2'] == team]
        
        home_wins = home_matches[home_matches['winner'] == team]
        away_wins = away_matches[away_matches['winner'] == team]
        
        home_win_rate = len(home_wins) / len(home_matches) if len(home_matches) > 0 else 0
        away_win_rate = len(away_wins) / len(away_matches) if len(away_matches) > 0 else 0
        
        # Store the features
        team_features[team] = {
            'total_matches': len(team_matches),
            'total_wins': len(team_wins),
            'win_rate': win_rate,
            'toss_win_rate': toss_win_rate,
            'home_win_rate': home_win_rate,
            'away_win_rate': away_win_rate
        }
    
    # Convert to DataFrame
    team_stats_df = pd.DataFrame.from_dict(team_features, orient='index')
    team_stats_df.reset_index(inplace=True)
    team_stats_df.rename(columns={'index': 'team'}, inplace=True)
    
    return team_stats_df

def create_venue_features(matches, deliveries):
    """Create venue-based features"""
    # Merge matches with deliveries to get venue info per delivery
    deliveries_with_venue = deliveries.merge(matches[['id', 'venue']], left_on='match_id', right_on='id')

    # Get all unique venues
    venues = matches['venue'].unique()
    venue_features = {}
    
    for venue in venues:
        if pd.isna(venue):
            continue
            
        # Matches at this venue
        venue_matches = matches[matches['venue'] == venue]
        
        # Team winning most at this venue
        venue_winners = venue_matches['winner'].value_counts()
        most_successful_team = venue_winners.idxmax() if not venue_winners.empty else None
        
        # Toss impact at venue
        toss_wins_match = venue_matches[venue_matches['toss_winner'] == venue_matches['winner']]
        toss_impact = len(toss_wins_match) / len(venue_matches) if len(venue_matches) > 0 else 0
        
        # Calculate Average first innings score using deliveries data
        # Filter deliveries for matches at this venue and first innings
        venue_deliveries = deliveries_with_venue[
            (deliveries_with_venue['venue'] == venue) & (deliveries_with_venue['inning'] == 1)
        ]
        # Group by match_id and sum total runs for the first innings
        first_innings_scores = venue_deliveries.groupby('match_id')['total_runs'].sum()
        # Calculate the average score
        avg_first_innings_score = first_innings_scores.mean() if not first_innings_scores.empty else 0
        
        # Store venue features
        venue_features[venue] = {
            'total_matches': len(venue_matches),
            'most_successful_team': most_successful_team,
            'toss_win_impact': toss_impact,
            'avg_first_innings_score': avg_first_innings_score
        }
    
    # Convert to DataFrame
    venue_stats_df = pd.DataFrame.from_dict(venue_features, orient='index')
    venue_stats_df.reset_index(inplace=True)
    venue_stats_df.rename(columns={'index': 'venue'}, inplace=True)
    
    return venue_stats_df

def create_match_momentum_features(matches):
    """Create match momentum features based on team's recent performance"""
    # Sort matches by date
    matches_sorted = matches.sort_values('date')
    
    # Create a copy to add features
    matches_momentum = matches_sorted.copy()
    
    # Initialize columns for momentum features
    matches_momentum['team1_last5_wins'] = 0
    matches_momentum['team2_last5_wins'] = 0
    matches_momentum['team1_form'] = 0  # Form as a momentum score
    matches_momentum['team2_form'] = 0
    
    # Process each match to build momentum features
    for idx, match in matches_sorted.iterrows():
        # Previous matches before this one
        prev_matches = matches_sorted[matches_sorted['date'] < match['date']]
        
        # Last 5 matches for each team
        team1_last5 = prev_matches[(prev_matches['team1'] == match['team1']) | 
                                  (prev_matches['team2'] == match['team1'])].tail(5)
        
        team2_last5 = prev_matches[(prev_matches['team1'] == match['team2']) | 
                                  (prev_matches['team2'] == match['team2'])].tail(5)
        
        # Count wins in last 5 matches
        team1_wins = len(team1_last5[team1_last5['winner'] == match['team1']])
        team2_wins = len(team2_last5[team2_last5['winner'] == match['team2']])
        
        # Calculate form (weighted by recency)
        team1_form = 0
        team2_form = 0
        
        for i, t1_match in enumerate(team1_last5.itertuples(), 1):
            weight = i / 15  # More recent matches have higher weight
            if t1_match.winner == match['team1']:
                team1_form += weight
        
        for i, t2_match in enumerate(team2_last5.itertuples(), 1):
            weight = i / 15
            if t2_match.winner == match['team2']:
                team2_form += weight
        
        # Update the features
        matches_momentum.at[idx, 'team1_last5_wins'] = team1_wins
        matches_momentum.at[idx, 'team2_last5_wins'] = team2_wins
        matches_momentum.at[idx, 'team1_form'] = team1_form
        matches_momentum.at[idx, 'team2_form'] = team2_form
    
    return matches_momentum

def create_head_to_head_features(matches):
    """Create head-to-head features between teams"""
    # Create dictionary to store head-to-head records
    h2h_records = {}
    matches_h2h = matches.copy()
    
    # Process matches to build h2h stats
    for _, match in matches.iterrows():
        team1, team2 = match['team1'], match['team2']
        winner = match['winner']
        
        # Create key for team pair
        team_pair = tuple(sorted([team1, team2]))
        
        if team_pair not in h2h_records:
            h2h_records[team_pair] = {'matches': 0, team1: 0, team2: 0}
        
        h2h_records[team_pair]['matches'] += 1
        
        if winner in [team1, team2]:  # Ensure winner is one of the teams (not no result/tie)
            h2h_records[team_pair][winner] += 1
    
    # Add head-to-head features to each match
    matches_h2h['team1_h2h_wins'] = 0
    matches_h2h['team2_h2h_wins'] = 0
    matches_h2h['team1_h2h_winrate'] = 0.5  # Default to 0.5 if no history
    matches_h2h['team2_h2h_winrate'] = 0.5
    
    for idx, match in matches.iterrows():
        team1, team2 = match['team1'], match['team2']
        team_pair = tuple(sorted([team1, team2]))
        
        if team_pair in h2h_records:
            h2h = h2h_records[team_pair]
            team1_wins = h2h.get(team1, 0)
            team2_wins = h2h.get(team2, 0)
            total_matches = h2h['matches']
            
            matches_h2h.at[idx, 'team1_h2h_wins'] = team1_wins
            matches_h2h.at[idx, 'team2_h2h_wins'] = team2_wins
            matches_h2h.at[idx, 'team1_h2h_winrate'] = team1_wins / total_matches if total_matches > 0 else 0.5
            matches_h2h.at[idx, 'team2_h2h_winrate'] = team2_wins / total_matches if total_matches > 0 else 0.5
    
    return matches_h2h

def create_player_features(matches, deliveries):
    """Create player performance features from ball-by-ball data"""
    # Merge deliveries with matches to get context
    deliveries_enriched = deliveries.merge(
        matches[['id', 'date', 'season', 'venue', 'team1', 'team2']], 
        left_on='match_id', 
        right_on='id'
    )
    
    # Batting statistics
    batsman_stats = deliveries.groupby('batter').agg(
        total_runs=('batsman_runs', 'sum'),
        balls_faced=('match_id', 'count'),
        fours=('batsman_runs', lambda x: sum(x == 4)),
        sixes=('batsman_runs', lambda x: sum(x == 6))
    ).reset_index()
    
    batsman_stats['strike_rate'] = (batsman_stats['total_runs'] / batsman_stats['balls_faced']) * 100
    
    # Bowling statistics
    bowler_stats = deliveries.groupby('bowler').agg(
        overs_bowled=('match_id', lambda x: len(x) / 6),  # Approximate overs
        wickets=('is_wicket', 'sum'),
        runs_conceded=('total_runs', 'sum')
    ).reset_index()
    
    bowler_stats['economy'] = bowler_stats['runs_conceded'] / bowler_stats['overs_bowled']
    
    # Create player-match level features - To be used when predicting player performance
    player_match_stats = deliveries_enriched.groupby(['match_id', 'batter']).agg(
        runs_scored=('batsman_runs', 'sum'),
        balls_faced=('match_id', 'count')
    ).reset_index()
    
    return batsman_stats, bowler_stats, player_match_stats

def calculate_streaks_before_match(matches):
    """Calculate win/loss streaks for each team before each match."""
    matches_sorted = matches.sort_values('date')
    team_streaks = {}
    team1_streaks = []
    team2_streaks = []

    for _, match in matches_sorted.iterrows():
        team1 = match['team1']
        team2 = match['team2']
        winner = match['winner']

        # Get current streak before this match
        t1_streak = team_streaks.get(team1, 0)
        t2_streak = team_streaks.get(team2, 0)
        team1_streaks.append(t1_streak)
        team2_streaks.append(t2_streak)

        # Update streak based on match result
        if winner == team1:
            team_streaks[team1] = t1_streak + 1 if t1_streak >= 0 else 1
            team_streaks[team2] = t2_streak - 1 if t2_streak <= 0 else -1
        elif winner == team2:
            team_streaks[team1] = t1_streak - 1 if t1_streak <= 0 else -1
            team_streaks[team2] = t2_streak + 1 if t2_streak >= 0 else 1
        else: # Handle ties or no results - reset streak? Or maintain?
              # Let's maintain for now, could be revisited.
              # Or perhaps reset to 0 if it's a tie/no result?
              # Resetting might be safer if these are rare.
              # For simplicity, let's reset on non-win/loss
              team_streaks[team1] = 0
              team_streaks[team2] = 0

    matches_sorted['team1_streak'] = team1_streaks
    matches_sorted['team2_streak'] = team2_streaks
    return matches_sorted.sort_index() # Return to original order

def calculate_venue_specific_team_stats(matches):
    """Calculate team win rates specifically at each venue."""
    matches_sorted = matches.sort_values('date')
    venue_team_stats = {}
    team1_venue_win_rates = []
    team2_venue_win_rates = []

    for _, match in matches_sorted.iterrows():
        venue = match['venue']
        team1 = match['team1']
        team2 = match['team2']
        winner = match['winner']

        # Initialize venue stats if not present
        if venue not in venue_team_stats:
            venue_team_stats[venue] = {}

        # Get current stats before this match
        t1_stats = venue_team_stats[venue].get(team1, {'played': 0, 'won': 0})
        t2_stats = venue_team_stats[venue].get(team2, {'played': 0, 'won': 0})

        # Calculate win rate before this match
        t1_venue_win_rate = t1_stats['won'] / t1_stats['played'] if t1_stats['played'] > 0 else 0.5 # Default 0.5 if no history
        t2_venue_win_rate = t2_stats['won'] / t2_stats['played'] if t2_stats['played'] > 0 else 0.5
        team1_venue_win_rates.append(t1_venue_win_rate)
        team2_venue_win_rates.append(t2_venue_win_rate)

        # Update stats based on match result
        # Update Team 1
        current_t1_stats = venue_team_stats[venue].get(team1, {'played': 0, 'won': 0})
        current_t1_stats['played'] += 1
        if winner == team1:
            current_t1_stats['won'] += 1
        venue_team_stats[venue][team1] = current_t1_stats

        # Update Team 2
        current_t2_stats = venue_team_stats[venue].get(team2, {'played': 0, 'won': 0})
        current_t2_stats['played'] += 1
        if winner == team2:
            current_t2_stats['won'] += 1
        venue_team_stats[venue][team2] = current_t2_stats

    matches_sorted['team1_venue_win_rate'] = team1_venue_win_rates
    matches_sorted['team2_venue_win_rate'] = team2_venue_win_rates
    return matches_sorted.sort_index() # Return to original order

def create_final_dataset(matches):
    """Combine all features into a final dataset"""
    # Start with the processed matches
    final_df = matches.copy()
    
    # Add match outcome
    final_df['team1_won'] = (final_df['team1'] == final_df['winner']).astype(int)
    
    # Create bat first decision feature
    final_df['chose_to_bat'] = ((final_df['toss_winner'] == final_df['team1']) & 
                              (final_df['toss_decision'] == 'bat')) | \
                             ((final_df['toss_winner'] == final_df['team2']) & 
                              (final_df['toss_decision'] == 'field'))
    final_df['chose_to_bat'] = final_df['chose_to_bat'].astype(int)
    
    # Add toss winner feature
    final_df['team1_won_toss'] = (final_df['toss_winner'] == final_df['team1']).astype(int)
    
    # Keep only relevant columns
    cols_to_keep = [
        'id', 'season', 'date', 'team1', 'team2', 'venue', 
        'toss_winner', 'toss_decision', 'chose_to_bat', 'team1_won_toss',
        'team1_last5_wins', 'team2_last5_wins', 'team1_form', 'team2_form',
        'team1_h2h_wins', 'team2_h2h_wins', 'team1_h2h_winrate', 'team2_h2h_winrate',
        'team1_streak', 'team2_streak',
        'team1_venue_win_rate', 'team2_venue_win_rate', # Added venue-specific win rates
        'winner', 'team1_won'
    ]
    
    # Filter columns that exist
    existing_cols = [col for col in cols_to_keep if col in final_df.columns]
    final_df = final_df[existing_cols]
    
    return final_df

def encode_categorical_features(df):
    """Encode categorical features for model training"""
    from sklearn.preprocessing import LabelEncoder
    
    df_encoded = df.copy()
    
    # Identify categorical columns (excluding target and identifiers)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['winner']]
    
    # Encode each categorical column
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle NaN values
        df_encoded[col] = df_encoded[col].fillna('Unknown')
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    return df_encoded

def main():
    """Main function to execute the feature engineering pipeline"""
    print("Starting feature engineering process...")
    
    # Load raw data
    matches, deliveries = load_data()
    
    print("Preprocessing matches data...")
    matches_processed = preprocess_matches(matches)
    
    print("Creating team features...")
    team_stats = create_team_features(matches_processed, deliveries)
    
    print("Creating venue features...")
    venue_stats = create_venue_features(matches_processed, deliveries)
    
    print("Creating match momentum features...")
    matches_with_momentum = create_match_momentum_features(matches_processed)

    print("Creating win/loss streak features...") # Added step
    matches_with_streaks = calculate_streaks_before_match(matches_with_momentum)
    
    print("Calculating venue-specific team win rates...") # Added step
    matches_with_venue_rates = calculate_venue_specific_team_stats(matches_with_streaks) # Use df with streaks

    print("Creating head-to-head features...")
    matches_with_h2h = create_head_to_head_features(matches_with_venue_rates) # Use df with venue rates
    
    print("Creating player features...")
    batsman_stats, bowler_stats, player_match_stats = create_player_features(matches_processed, deliveries)
    
    print("Creating final dataset...")
    final_dataset = create_final_dataset(matches_with_h2h)
    
    print("Encoding categorical features...")
    final_dataset_encoded = encode_categorical_features(final_dataset)
    
    # Save processed data
    print("Saving processed data...")
    team_stats.to_csv(os.path.join(PROCESSED_DATA_DIR, 'team_stats.csv'), index=False)
    venue_stats.to_csv(os.path.join(PROCESSED_DATA_DIR, 'venue_stats.csv'), index=False)
    batsman_stats.to_csv(os.path.join(PROCESSED_DATA_DIR, 'batsman_stats.csv'), index=False)
    bowler_stats.to_csv(os.path.join(PROCESSED_DATA_DIR, 'bowler_stats.csv'), index=False)
    player_match_stats.to_csv(os.path.join(PROCESSED_DATA_DIR, 'player_match_stats.csv'), index=False)
    final_dataset.to_csv(os.path.join(PROCESSED_DATA_DIR, 'matches_features.csv'), index=False)
    final_dataset_encoded.to_csv(os.path.join(PROCESSED_DATA_DIR, 'matches_features_encoded.csv'), index=False)
    
    print(f"Feature engineering complete. Processed data saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()