# 🏏 IPL Match Outcome Predictor

An advanced AI-based system for predicting IPL match outcomes, final scores, key player performance, and explaining predictions using a hybrid of machine learning and a local large language model (LLM) via Ollama.

---

## 🔧 Tech Stack

| Component        | Technology                         |
|------------------|-------------------------------------|
| ML Models        | Scikit-learn, XGBoost, numpy              |
| Backend APIs     | FastAPI, Django                    |
| LLM Integration  | Gemini 2.0     |
| Data Pipelines   | Pandas, NumPy, Joblib              |
| Visualization    | Django Admin, matplotlib                      |
| Live API Docs    | Swagger (FastAPI built-in)         |
| Deployment       | Uvicorn, Docker                           |

---

## 📁 Project Structure

```
ipl_predictor/
├── data/                   # IPL datasets (historical match & player data)
│   ├── processed/          # Processed CSV files after feature engineering
│   └── raw/                # Raw match and deliveries data
├── ipl_backend/           # Django admin + visualization dashboard
│   ├── predictor_ui/      # Django app for UI interface
│   └── ...
├── ml-models/              # Serialized ML models
│   ├── expected_columns.joblib
│   ├── ipl_predictor_rf_classifier_tuned.joblib
│   └── ...
├── notebooks/              # Exploratory Data Analysis & model training
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
├── ollama-llm/             # LLM integration
│   └── models.py
├── scripts/                # Data processing and model training scripts
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── ...
├── fast_api.py            # FastAPI server implementation
├── run_servers.py         # Script to run both Django and FastAPI servers
└── requirements.txt       # Project dependencies
```

---

## 🚀 Features

### 🧠 ML Predictions

- Predict match winner with probability score
- Team performance metrics including win rates and averages
- Feature importance for prediction explanation
- Multiple model implementations (Random Forest, XGBoost)

### 💬 LLM Reasoning (Ollama)

- Local LLM for explaining predictions in natural language
- Analysis of match stats and historical performance

### 📊 Match Data Analysis

- Team stats: Win rates, average runs, economy rates
- Player performance metrics
- Venue-specific analysis

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ipl-predictor.git
cd ipl-predictor
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv env
# On Windows
env\Scripts\activate
# On macOS/Linux
source env/bin/activate
```

### 3. Install Python Requirements

```bash
pip install -r requirements.txt
```

### 4. Start FastAPI Server

```bash
python fast_api.py
```

The API will be available at http://127.0.0.1:8000 with Swagger documentation at http://127.0.0.1:8000/docs

### 5. Start Django Backend (Optional)

```bash
cd ipl_backend
python manage.py runserver
```

### 6. Run Ollama LLM (Locally)

Install [Ollama](https://ollama.com/) and run:

```bash
ollama run llama2
```

> Ensure your FastAPI or Django backend communicates with Ollama via HTTP (default: `http://localhost:11434`).

---

## 🔗 API Endpoints (FastAPI)

| Method | Endpoint     | Description                      |
|--------|-------------|----------------------------------|
| POST   | `/predict`  | Predict match winner with probability |

---

## 📊 Sample Prediction Input (JSON)

```json
{
  "venue": "Wankhede Stadium",
  "team1": "Chennai Super Kings",
  "team2": "Mumbai Indians",
  "toss_winner": "Chennai Super Kings",
  "toss_decision": "bat",
  "team1_win_rate_last_5": 0.6,
  "team1_avg_margin_last_5": 15.0,
  "team2_win_rate_last_5": 0.4,
  "team2_avg_margin_last_5": -10.0,
  "team1_avg_runs_scored_last_5": 175.2,
  "team1_avg_wickets_taken_last_5": 6.8,
  "team1_avg_economy_rate_last_5": 7.8,
  "team2_avg_runs_scored_last_5": 165.5,
  "team2_avg_wickets_taken_last_5": 6.2,
  "team2_avg_economy_rate_last_5": 8.2
}
```

---

## 🧠 Model Training Pipeline

The project includes a comprehensive data processing and model training pipeline:

1. **Data Processing** (`scripts/data_cleaning.py`, `scripts/feature_engineering.py`)
   - Clean and merge raw match and delivery data
   - Generate features for team, player, and venue performance

2. **Model Training** (`scripts/model_training.py`, `scripts/hyper_tuning.py`)
   - Train Random Forest and XGBoost classifiers
   - Hyperparameter tuning for optimized performance
   - Feature selection and importance analysis

3. **Model Evaluation** (`notebooks/model_training.ipynb`)
   - Performance metrics (accuracy, precision, recall, F1)
   - Cross-validation results
   - Feature importance visualization

---

## 🔄 Running Both Servers

Use the `run_servers.py` script to start both Django and FastAPI servers simultaneously:

```bash
python run_servers.py
```

---

## 📱 Web UI Access

Once the Django server is running, access the web UI at:
- http://127.0.0.1:8000/

The admin interface is available at:
- http://127.0.0.1:8000/admin/

---

## 📜 License

**MIT License** – Free to use, modify, and distribute.

---

## 🙌 Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

---

## 👨‍💻 Author

**ANUBHOB DEY**  
AI Engineer & Backend Developer
