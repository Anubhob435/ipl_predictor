Great! Let’s walk step-by-step through **✅ Step 2: Data Collection & Feature Engineering** to build a structured dataset and prepare it for training ML models to predict IPL match outcomes.

---

## ✅ Step 2: Data Collection & Feature Engineering

---

### 🎯 **Goal Recap:**

* Collect historical IPL data
* Build meaningful features: teams’ recent form, player performance, match context (venue, toss, etc.)
* Make it reproducible with tools like `sklearn.Pipeline` or `DVC`

---

## 🔹 Step-by-Step Breakdown

---

### 🔸 **1. Download Historical IPL Match Data**

You can get datasets from:

* [Kaggle IPL Datasets](https://www.kaggle.com/datasets)
* Cricinfo/Cricsheet (for ball-by-ball level data)
* CSV or JSON formats preferred

**Example dataset files:**

* `matches.csv` (match-level data)
* `deliveries.csv` (ball-by-ball data)

---

### 🔸 **2. Load Data with Pandas**

```python
import pandas as pd

matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')
```

---

### 🔸 **3. Clean and Preprocess Data**

Steps:

* Convert dates to datetime format
* Fill or drop missing values
* Normalize team names (if inconsistent)
* Remove abandoned/no result matches

```python
matches['date'] = pd.to_datetime(matches['date'])
matches = matches[matches['result'] != 'no result']
```

---

### 🔸 **4. Engineer Team-Based Features**

Focus on the **last 3–5 matches per team**:

#### a. **Team Averages (Last 5 Games)**

* Runs scored
* Wickets taken
* Runs conceded

```python
def team_last_n_games_avg(df, team, n=5):
    team_matches = df[(df['team1'] == team) | (df['team2'] == team)].sort_values('date', ascending=False)
    last_n = team_matches.head(n)
    avg_runs = last_n['team1_runs'].mean()  # example
    return avg_runs
```

---

#### b. **Momentum (Win/Loss Streak)**

Create a column to track team win streaks:

```python
matches['winner_prev'] = matches.groupby('team1')['winner'].shift(1)
matches['win_streak'] = matches.groupby('team1')['winner'].apply(lambda x: x.eq(x.shift()).cumsum())
```

---

### 🔸 **5. Engineer Player Form Metrics**

From `deliveries.csv`:

* Batting average over last 5 innings
* Bowling economy rate
* Recent wickets taken

You’ll need to:

* Group by player
* Sort by match date
* Aggregate over last N matches

Example for batsman:

```python
def player_form(deliveries, player_name, recent_matches=5):
    player_deliveries = deliveries[deliveries['batsman'] == player_name]
    recent = player_deliveries.sort_values('match_id', ascending=False).head(recent_matches)
    total_runs = recent['batsman_runs'].sum()
    return total_runs / recent_matches
```

---

### 🔸 **6. Add Match Context Features**

#### a. **Toss Winner Advantage**

```python
matches['toss_win_match_win'] = (matches['toss_winner'] == matches['winner']).astype(int)
```

#### b. **Venue Advantage**

Check if the team is playing in its home city:

```python
matches['is_home_team'] = matches.apply(lambda row: row['team1'] in row['venue'], axis=1)
```

---

### 🔸 **7. Build Feature Pipelines with sklearn**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
processed_features = pipeline.fit_transform(X)
```

---

### 🔸 **8. Make It Reproducible with DVC (Optional)**

If you want full reproducibility with versioned data and models:

* Install DVC:

  ```bash
  pip install dvc
  dvc init
  ```

* Track your data:

  ```bash
  dvc add data/matches.csv
  git add data/matches.csv.dvc
  git commit -m "Track IPL match data with DVC"
  ```

* Define data processing in a DVC stage:

  ```bash
  dvc run -n preprocess \
    -d src/feature_engineering.py \
    -o data/processed.csv \
    python src/feature_engineering.py
  ```

---

### 📦 Output

At the end of this step, you'll have:

* A cleaned, feature-rich DataFrame (`X`)
* A target column (`y`) like match winner or score
* A reproducible pipeline ready for ML modeling

---

Would you like a sample `feature_engineering.py` file or a Jupyter Notebook template for this?
