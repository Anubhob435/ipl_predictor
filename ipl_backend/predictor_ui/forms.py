from django import forms

class PredictionForm(forms.Form):
    # Basic match information fields
    venue = forms.CharField(
        label='Venue',
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    team1 = forms.CharField(
        label='Team 1',
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    team2 = forms.CharField(
        label='Team 2',
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    toss_winner = forms.CharField(
        label='Toss Winner',
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    toss_decision = forms.ChoiceField(
        label='Toss Decision',
        choices=[('bat', 'Bat'), ('field', 'Field')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    # Team 1 performance stats
    team1_win_rate_last_5 = forms.FloatField(
        label='Team 1 Win Rate (Last 5 matches)',
        min_value=0.0,
        max_value=1.0,
        initial=0.5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'})
    )
    team1_avg_margin_last_5 = forms.FloatField(
        label='Team 1 Average Margin (Last 5 matches)',
        initial=0.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'})
    )
    team1_avg_runs_scored_last_5 = forms.FloatField(
        label='Team 1 Average Runs Scored (Last 5 matches)',
        min_value=0.0,
        initial=150.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'})
    )
    team1_avg_wickets_taken_last_5 = forms.FloatField(
        label='Team 1 Average Wickets Taken (Last 5 matches)',
        min_value=0.0,
        max_value=10.0,
        initial=5.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'})
    )
    team1_avg_economy_rate_last_5 = forms.FloatField(
        label='Team 1 Average Economy Rate (Last 5 matches)',
        min_value=0.0,
        initial=8.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'})
    )
    
    # Team 2 performance stats
    team2_win_rate_last_5 = forms.FloatField(
        label='Team 2 Win Rate (Last 5 matches)',
        min_value=0.0,
        max_value=1.0,
        initial=0.5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'})
    )
    team2_avg_margin_last_5 = forms.FloatField(
        label='Team 2 Average Margin (Last 5 matches)',
        initial=0.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'})
    )
    team2_avg_runs_scored_last_5 = forms.FloatField(
        label='Team 2 Average Runs Scored (Last 5 matches)',
        min_value=0.0,
        initial=150.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'})
    )
    team2_avg_wickets_taken_last_5 = forms.FloatField(
        label='Team 2 Average Wickets Taken (Last 5 matches)',
        min_value=0.0,
        max_value=10.0,
        initial=5.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'})
    )
    team2_avg_economy_rate_last_5 = forms.FloatField(
        label='Team 2 Average Economy Rate (Last 5 matches)',
        min_value=0.0,
        initial=8.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'})
    )