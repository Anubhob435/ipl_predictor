import pandas as pd
import os

# Define paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

# Create processed data directory if it doesn't exist
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

def team_performance_over_last_n_games(matches_df, deliveries_df, team, n=5):
    # Filter matches for the given team
    team_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)]
    
    # Sort by date and keep the last 'n' matches
    team_matches = team_matches.sort_values(by='date', ascending=False).head(n)
    
    # Merge deliveries data to get performance details
    team_deliveries = deliveries_df[deliveries_df['match_id'].isin(team_matches['id'])]
    
    # Calculate the average runs scored by the team
    total_runs = team_deliveries.groupby('batsman')['runs'].sum().sum()
    average_runs = total_runs / len(team_matches)
    
    # Return the feature: average runs scored in last 'n' matches
    return average_runs

def player_batting_average(deliveries_df, player):
    # Filter deliveries for the given player
    player_deliveries = deliveries_df[deliveries_df['batsman'] == player]
    
    # Calculate batting average (total runs / total innings played)
    total_runs = player_deliveries['runs'].sum()
    total_innings = player_deliveries['match_id'].nunique()
    
    batting_average = total_runs / total_innings if total_innings > 0 else 0
    return batting_average

def calculate_win_streak(matches_df, team):
    """Calculate the current win streak (positive) or loss streak (negative) for a team."""
    # Get matches where the team played, in chronological order
    team_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)].sort_values('date')
        
    streak = 0
    for _, match in team_matches.iterrows():
        if match['winner'] == team:
            streak = streak + 1 if streak >= 0 else 1
        else:
            streak = streak - 1 if streak <= 0 else -1
        
    return streak

def player_win_streak(matches_df, deliveries_df, player):
    """Calculate win streak for a player based on their team's performance."""
    # Get matches where the player participated
    player_matches = matches_df[matches_df['id'].isin(
        deliveries_df[deliveries_df['batsman'] == player]['match_id'].unique()
    )].sort_values('date')
        
    streak = 0
    for _, match in player_matches.iterrows():
        # Find player's team for this match from deliveries
        match_deliveries = deliveries_df[deliveries_df['match_id'] == match['id']]
        player_team = match_deliveries[match_deliveries['batsman'] == player]['batting_team'].iloc[0]
            
        if match['winner'] == player_team:
            streak = streak + 1 if streak >= 0 else 1
        else:
            streak = streak - 1 if streak <= 0 else -1
        
    return streak

def calculate_recent_team_performance(matches_df, team_stats_df, n=5):
    """Calculates rolling win rate, avg margin, avg runs scored, avg wickets taken, avg economy rate for each team."""
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    matches_df = matches_df.sort_values('date')

    # Merge match-level team stats into matches_df for easier lookup
    # We need stats for both team1 and team2 perspective in each match row
    team_stats_df_t1 = team_stats_df.rename(columns={'team': 'team1', 'runs_scored': 't1_runs', 'wickets_taken': 't1_wickets', 'economy_rate': 't1_econ'})
    team_stats_df_t2 = team_stats_df.rename(columns={'team': 'team2', 'runs_scored': 't2_runs', 'wickets_taken': 't2_wickets', 'economy_rate': 't2_econ'})
    
    # Ensure the merge columns have compatible types if necessary (assuming 'id' is int and 'match_id' is int)
    # matches_df['id'] = matches_df['id'].astype(int) # Already done earlier
    # team_stats_df_t1['match_id'] = team_stats_df_t1['match_id'].astype(int) # Already done earlier
    # team_stats_df_t2['match_id'] = team_stats_df_t2['match_id'].astype(int) # Already done earlier

    # Correct merge using left_on and right_on
    matches_df = pd.merge(matches_df, team_stats_df_t1[['match_id', 't1_runs', 't1_wickets', 't1_econ']], left_on='id', right_on='match_id', how='left')
    matches_df = pd.merge(matches_df, team_stats_df_t2[['match_id', 't2_runs', 't2_wickets', 't2_econ']], left_on='id', right_on='match_id', how='left')
    
    # Drop the redundant match_id columns from the merges
    matches_df.drop(columns=['match_id_x', 'match_id_y'], inplace=True, errors='ignore')


    teams = pd.concat([matches_df['team1'], matches_df['team2']]).unique()
    new_features = []

    for index, match in matches_df.iterrows():
        match_date = match['date']
        team1 = match['team1']
        team2 = match['team2']
        match_features = {'id': match['id']}

        for team, prefix in [(team1, 'team1'), (team2, 'team2')]:
            # Filter past matches for the current team before the current match date
            past_matches = matches_df[((matches_df['team1'] == team) | (matches_df['team2'] == team)) & (matches_df['date'] < match_date)]
            last_n_matches = past_matches.tail(n)

            if len(last_n_matches) > 0:
                # Win Rate
                wins = (last_n_matches['winner'] == team).sum()
                win_rate = wins / len(last_n_matches)

                # Avg Margin
                margins = []
                for _, past_match in last_n_matches.iterrows():
                    margin = past_match['result_margin']
                    if pd.isna(margin): margin = 0
                    if past_match['winner'] != team: margin = -margin
                    margins.append(margin)
                avg_margin = sum(margins) / len(margins) if margins else 0

                # Avg Runs Scored, Wickets Taken, Economy Rate
                runs_scored_list = []
                wickets_taken_list = []
                economy_rate_list = []
                for _, past_match in last_n_matches.iterrows():
                    if past_match['team1'] == team:
                        runs_scored_list.append(past_match['t1_runs'])
                        wickets_taken_list.append(past_match['t1_wickets']) # Wickets taken *by* team1
                        economy_rate_list.append(past_match['t1_econ']) # Economy rate *of* team1 bowlers
                    elif past_match['team2'] == team:
                        runs_scored_list.append(past_match['t2_runs'])
                        wickets_taken_list.append(past_match['t2_wickets']) # Wickets taken *by* team2
                        economy_rate_list.append(past_match['t2_econ']) # Economy rate *of* team2 bowlers

                avg_runs_scored = pd.Series(runs_scored_list).mean()
                avg_wickets_taken = pd.Series(wickets_taken_list).mean()
                avg_economy_rate = pd.Series(economy_rate_list).mean() # Mean ignores NA by default

            else:
                win_rate = 0.0
                avg_margin = 0.0
                avg_runs_scored = 0.0
                avg_wickets_taken = 0.0
                avg_economy_rate = pd.NA # Use NA for unknown avg economy

            match_features[f'{prefix}_win_rate_last_{n}'] = win_rate
            match_features[f'{prefix}_avg_margin_last_{n}'] = avg_margin
            match_features[f'{prefix}_avg_runs_scored_last_{n}'] = avg_runs_scored
            match_features[f'{prefix}_avg_wickets_taken_last_{n}'] = avg_wickets_taken
            match_features[f'{prefix}_avg_economy_rate_last_{n}'] = avg_economy_rate

        new_features.append(match_features)

    new_features_df = pd.DataFrame(new_features)
    # Fill NA economy rates with a default/median if desired, or handle later
    # For now, let's fill with a relatively high value like 10, assuming missing means poor bowling performance or not bowled
    # Fix FutureWarning: Use assignment instead of inplace=True
    fill_values = {col: 10.0 for col in new_features_df.columns if 'economy_rate' in col}
    new_features_df = new_features_df.fillna(value=fill_values)

    matches_df = pd.merge(matches_df, new_features_df, on='id', how='left')
    # Drop the temporary match-level stats columns
    matches_df.drop(columns=['t1_runs', 't1_wickets', 't1_econ', 't2_runs', 't2_wickets', 't2_econ'], inplace=True, errors='ignore')
    return matches_df

def calculate_inning_scores(deliveries_df):
    """Calculates the total score for each inning of each match."""
    # Filter for only the first two innings
    main_innings_df = deliveries_df[deliveries_df['inning'].isin([1, 2])]
    inning_scores = main_innings_df.groupby(['match_id', 'inning'])['total_runs'].sum()

    # Use unstack to pivot innings into columns, fill missing innings with 0
    inning_scores_unstacked = inning_scores.unstack(fill_value=0).reset_index()

    # Rename columns explicitly: match_id becomes id, inning numbers become score columns
    inning_scores_unstacked = inning_scores_unstacked.rename(columns={
        'match_id': 'id',
        1: 'inning1_score',
        2: 'inning2_score'
    })

    # Ensure both inning score columns exist, adding inning2_score if missing
    # (e.g., match abandoned after 1st inning, though unstack(fill_value=0) should handle this)
    if 'inning1_score' not in inning_scores_unstacked.columns:
        inning_scores_unstacked['inning1_score'] = 0
    if 'inning2_score' not in inning_scores_unstacked.columns:
        inning_scores_unstacked['inning2_score'] = 0

    # Select and reorder columns to ensure consistency
    inning_scores_final = inning_scores_unstacked[['id', 'inning1_score', 'inning2_score']]

    return inning_scores_final

# --- New Function to Calculate Detailed Team Stats per Match ---
def calculate_team_stats_per_match(deliveries_df):
    """Calculates runs scored, wickets taken, balls bowled, and economy rate per team per match."""
    # Runs scored by each team in each match
    runs_scored = deliveries_df.groupby(['match_id', 'batting_team'])['total_runs'].sum().reset_index()
    runs_scored.rename(columns={'total_runs': 'runs_scored', 'batting_team': 'team'}, inplace=True)

    # Wickets taken by each team in each match (excluding run-outs, etc., where bowler is NaN)
    wickets_taken = deliveries_df.dropna(subset=['bowler']).groupby(['match_id', 'bowling_team'])['player_dismissed'].count().reset_index()
    wickets_taken.rename(columns={'player_dismissed': 'wickets_taken', 'bowling_team': 'team'}, inplace=True)

    # Balls bowled and runs conceded by each team in each match
    bowling_stats = deliveries_df.groupby(['match_id', 'bowling_team']).agg(
        runs_conceded=('total_runs', 'sum'),
        balls_bowled=('ball', 'count') # Count deliveries as balls bowled
    ).reset_index()
    # Adjust balls bowled for wides and noballs (they count as runs conceded but not as legal deliveries for economy)
    # Check if columns exist before using them
    conditions = []
    if 'wide_runs' in deliveries_df.columns:
        conditions.append(deliveries_df['wide_runs'] > 0)
    if 'noball_runs' in deliveries_df.columns:
        conditions.append(deliveries_df['noball_runs'] > 0)

    if conditions:
        # Combine conditions with logical OR
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition | condition
        
        extras_df = deliveries_df[combined_condition]
        extras_count = extras_df.groupby(['match_id', 'bowling_team']).size().reset_index(name='extra_balls')
        bowling_stats = pd.merge(bowling_stats, extras_count, on=['match_id', 'bowling_team'], how='left')
    else:
        # If neither column exists, add extra_balls column with zeros
        bowling_stats['extra_balls'] = 0
        
    # Fix FutureWarning: Use assignment instead of inplace=True
    bowling_stats['extra_balls'] = bowling_stats['extra_balls'].fillna(0) # Fill NaNs if merge didn't find extras for some teams
    bowling_stats['legal_balls_bowled'] = bowling_stats['balls_bowled'] - bowling_stats['extra_balls']
    # Calculate Economy Rate = (Runs Conceded / (Legal Balls Bowled / 6))
    # Avoid division by zero if no legal balls were bowled
    bowling_stats['economy_rate'] = (bowling_stats['runs_conceded'] * 6) / bowling_stats['legal_balls_bowled']
    bowling_stats['economy_rate'] = bowling_stats['economy_rate'].replace([float('inf'), -float('inf')], pd.NA) # Handle division by zero -> NA
    bowling_stats.rename(columns={'bowling_team': 'team'}, inplace=True)

    # Merge all stats
    team_stats = pd.merge(runs_scored, wickets_taken, on=['match_id', 'team'], how='outer')
    team_stats = pd.merge(team_stats, bowling_stats[['match_id', 'team', 'runs_conceded', 'legal_balls_bowled', 'economy_rate']], on=['match_id', 'team'], how='outer')

    # Fill NaNs that logically should be 0 (e.g., 0 wickets taken if none recorded)
    team_stats.fillna({'runs_scored': 0, 'wickets_taken': 0, 'runs_conceded': 0, 'legal_balls_bowled': 0}, inplace=True)
    # Keep economy_rate as NA if it couldn't be calculated

    return team_stats

if __name__ == "__main__":
    print("Loading raw match and delivery data...")
    matches_path = os.path.join(RAW_DATA_DIR, 'matches.csv')
    deliveries_path = os.path.join(RAW_DATA_DIR, 'deliveries.csv')
    matches_raw = pd.read_csv(matches_path)
    deliveries_raw = pd.read_csv(deliveries_path)

    # --- Data Cleaning (Minimal) ---
    matches_raw.replace('Rising Pune Supergiant', 'Rising Pune Supergiants', inplace=True)
    matches_raw.replace('Delhi Daredevils', 'Delhi Capitals', inplace=True)
    # Explicitly create a copy after dropna to avoid SettingWithCopyWarning
    matches_cleaned = matches_raw.dropna(subset=['winner']).copy()
    matches_cleaned['id'] = matches_cleaned['id'].astype(int) # Ensure id is int for merging
    deliveries_raw['match_id'] = deliveries_raw['match_id'].astype(int) # Ensure id is int for merging

    print("Calculating inning scores...")
    inning_scores_df = calculate_inning_scores(deliveries_raw)

    # Merge inning scores with matches
    matches_with_scores = pd.merge(matches_cleaned, inning_scores_df, on='id', how='left')
    # Fill potential missing scores after merge using assignment (avoids FutureWarning)
    matches_with_scores['inning1_score'] = matches_with_scores['inning1_score'].fillna(0)
    matches_with_scores['inning2_score'] = matches_with_scores['inning2_score'].fillna(0)

    # --- Calculate Detailed Team Stats Per Match ---
    print("Calculating detailed team stats per match...")
    team_stats_per_match_df = calculate_team_stats_per_match(deliveries_raw)
    # Ensure match IDs are compatible for merging (should already be int)
    team_stats_per_match_df['match_id'] = team_stats_per_match_df['match_id'].astype(int)


    print("Calculating recent performance features...")
    # Pass both matches and the new team stats df to the function
    # Ensure 'id' in matches_with_scores is int before passing
    matches_with_scores['id'] = matches_with_scores['id'].astype(int) 
    matches_with_features = calculate_recent_team_performance(matches_with_scores.copy(), team_stats_per_match_df.copy(), n=5)

    # Determine winning and losing scores
    def get_winning_losing_scores(row):
        if pd.isna(row['winner']):
            return pd.NA, pd.NA
        # Assuming inning 1 team is team1, inning 2 team is team2 for simplicity
        # This might need refinement based on actual batting order if not consistent
        score1 = row['inning1_score']
        score2 = row['inning2_score']
        if row['winner'] == row['team1']:
            # Team 1 won
            if row['result'] == 'wickets': # Team 2 batted first
                return score2, score1
            else: # Team 1 batted first
                return score1, score2
        elif row['winner'] == row['team2']:
            # Team 2 won
            if row['result'] == 'wickets': # Team 1 batted first
                return score1, score2
            else: # Team 2 batted first
                return score2, score1
        else: # Tie or No Result
             # Simple assignment for now, might need better logic for ties
             return max(score1, score2), min(score1, score2)

    scores = matches_with_features.apply(get_winning_losing_scores, axis=1, result_type='expand')
    matches_with_features[['winning_score', 'losing_score']] = scores

    # --- Add other features if needed ---
    # Example: Add toss advantage feature
    matches_with_features['toss_winner_is_team1'] = (matches_with_features['toss_winner'] == matches_with_features['team1']).astype(int)
    matches_with_features['team1_bat_first'] = (matches_with_features['toss_decision'] == 'bat').astype(int)
    matches_with_features['team1_toss_bat_first'] = matches_with_features['toss_winner_is_team1'] * matches_with_features['team1_bat_first']
    matches_with_features['team2_toss_bat_first'] = (1 - matches_with_features['toss_winner_is_team1']) * matches_with_features['team1_bat_first']

    # Define target variable
    matches_with_features['team1_won'] = (matches_with_features['winner'] == matches_with_features['team1']).astype(int)

    # --- Select and Save Features ---
    # Select relevant columns (add the new features)
    features_to_keep = [
        'id', 'date', 'team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner',
        'inning1_score', 'inning2_score', # Raw scores
        'winning_score', 'losing_score', # Target scores
        # Existing rolling features
        'team1_win_rate_last_5', 'team1_avg_margin_last_5',
        'team2_win_rate_last_5', 'team2_avg_margin_last_5',
        # New rolling features
        'team1_avg_runs_scored_last_5', 'team1_avg_wickets_taken_last_5', 'team1_avg_economy_rate_last_5',
        'team2_avg_runs_scored_last_5', 'team2_avg_wickets_taken_last_5', 'team2_avg_economy_rate_last_5',
        # Other features
        'toss_winner_is_team1', 'team1_bat_first',
        'team1_toss_bat_first', 'team2_toss_bat_first',
        'team1_won' # Target variable
    ]
    # Ensure only existing columns are selected
    # Add .copy() to avoid SettingWithCopyWarning on the subsequent dropna
    final_features = matches_with_features[[col for col in features_to_keep if col in matches_with_features.columns]].copy()

    # Drop rows where scores couldn't be determined (if any)
    final_features.dropna(subset=['winning_score', 'losing_score'], inplace=True)

    output_path = os.path.join(PROCESSED_DATA_DIR, 'matches_features.csv')
    final_features.to_csv(output_path, index=False)
    print(f"Feature engineered data (including scores and recent stats) saved to {output_path}") # Updated print message

    # --- Encoding Step (Placeholder - Needs to be run after this script) ---
    # The encoding logic that created 'matches_features_encoded.csv' 
    # should now run on 'matches_features.csv'.
    # This might involve one-hot encoding 'venue', 'team1', 'team2', etc.
    print("\nNOTE: Encoding step needs to be performed on the new 'matches_features.csv' file.")
    print("The 'model_training.py' script should then load the newly encoded file.")