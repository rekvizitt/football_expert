import datetime
import pandas as pd
from src.api import FootballExpertApi
from src.logger import logger

class BayesianNetwork:
    def __init__(self):
        # Разница в рейтингах (Общий рейтинг) (A) Положительная (хозяева выше) Отрицательная (гости выше)
        self.rating_diff = {"name": "A", "+": 0.0, "-": 0.0}
        # Разница в результативности (Забитые - пропущенные) (B) Положительная (хозяева результативнее) Отрицательная (гости результативнее)
        self.perfomance_diff = {"name": "B", "+": 0.0, "-": 0.0}
        # Разница в форме (Победы за последние 5 матчей) (C) Положительная (хозяева в лучшей форме) Отрицательная (гости в лучшей форме)
        self.form_diff = {"name": "C", "+": 0.0, "-": 0.0}
        # Преимущество домашнего поля (Владение мяча дома) (D) Есть (>50% владения) Нет (≤50% владения)
        self.home_advantage = {"name": "D", "+": 0.0, "-": 0.0}
        # Результат матча (победа хозяев, ничья, победа гостей) (E)
        self.match_result = {"name": "E", "home_win": 0.0, "draw": 0.0, "away_win": 0.0}
                
    def _calc_rating_diff(self, matches):
        for _, match in matches.iterrows():
            if match["home_overall_rating"] >= match["away_overall_rating"]:
                self.rating_diff["+"] += 1.0
            else:
                self.rating_diff["-"] += 1.0
        self.rating_diff["+"] /= len(matches)
        self.rating_diff["-"] /= len(matches)
                
    def _calc_perfomance_diff(self, matches):
        for _, match in matches.iterrows():
            home_perfomance = match["home_goals_last_5"] - match["home_conceded_goals_last_5"]
            away_perfomance = match["away_goals_last_5"] - match["away_conceded_goals_last_5"]
            if home_perfomance >= away_perfomance:
                self.perfomance_diff["+"] += 1.0
            else:
                self.perfomance_diff["-"] += 1.0
        self.perfomance_diff["+"] /= len(matches)
        self.perfomance_diff["-"] /= len(matches)

    def _calc_form_diff(self, matches):
        for _, match in matches.iterrows():
            if match["home_wins_last_5"] >= match["away_wins_last_5"]:
                self.form_diff["+"] += 1.0
            else:
                self.form_diff["-"] += 1.0
        self.form_diff["+"] /= len(matches)
        self.form_diff["-"] /= len(matches)
                
    def _calc_home_advantage(self, matches):
        for _, match in matches.iterrows():
            if match["home_possession"] > 50.0:
                self.home_advantage["+"] += 1.0
            else:
                self.home_advantage["-"] += 1.0
        self.home_advantage["+"] /= len(matches)
        self.home_advantage["-"] /= len(matches)
   
    def calc_factors(self, matches):
        self._calc_rating_diff(matches)
        self._calc_perfomance_diff(matches)
        self._calc_form_diff(matches)
        self._calc_home_advantage(matches)
   
    def calc_match_result(self, match):    
        A = '+' if match["home_overall_rating"] >= match["away_overall_rating"] else '-'
        B = '+' if (match["home_goals_last_5"] - match["home_conceded_goals_last_5"]) >= (match["away_goals_last_5"] - match["away_conceded_goals_last_5"]) else '-'
        C = '+' if match["home_wins_last_5"] >= match["away_wins_last_5"] else '-'
        D = '+' if match["home_possession"] > 50.0 else '-'  
        
        P_home_win_given_factors = (
            self.rating_diff[A] *
            self.perfomance_diff[B] *
            self.form_diff[C] *
            self.home_advantage[D]
        )

        P_away_win_given_factors = (
            self.rating_diff['-' if A == '+' else '+'] *
            self.perfomance_diff['-' if B == '+' else '+'] *
            self.form_diff['-' if C == '+' else '+'] *
            self.home_advantage['-' if D == '+' else '+']
        )
        
        P_draw_given_factors = 1 - P_home_win_given_factors + P_away_win_given_factors
        
        total_prob = P_home_win_given_factors + P_draw_given_factors + P_away_win_given_factors
        self.match_result["home_win"] = P_home_win_given_factors / total_prob if total_prob > 0 else 0
        self.match_result["draw"] = P_draw_given_factors / total_prob if total_prob > 0 else 0
        self.match_result["away_win"] = P_away_win_given_factors / total_prob if total_prob > 0 else 0

        return match

if __name__ == "__main__":
    leagues = ["ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", "GER-Bundesliga", "ITA-Serie A"]
    seasons = ["2425"]
    api = FootballExpertApi(leagues, seasons)
    api.dp.get_train_data()
    # ~1520 matches
    matches = api.dp.train_data
    
    bn = BayesianNetwork()
    # Данные для таблиц 1-4.
    bn.calc_factors(matches)
    logger.debug(f"Фактор А: {bn.rating_diff}, Фактор B: {bn.perfomance_diff}, Фактор C: {bn.form_diff}, Фактор D: {bn.home_advantage}")
    date = datetime.datetime(2025, 3, 1)
    # Данные для таблицы 5.
    matches = [
    # A+ B+ C+ D+
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 10, 'home_conceded_goals_last_5': 5,
        'home_xg_last_5': 9.5, 'home_wins_last_5': 4,
        'home_draws_last_5': 1, 'home_losses_last_5': 0,
        'home_possession': 60, 'home_ga_per_90': 0.8,
        'home_performance_ga': 1.2, 'home_xg': 2.1,
        'home_xag': 2.3, 'home_prgc': 12, 'home_prgp': 8,
        'home_attack_rating': 82, 'home_midfield_rating': 81,
        'home_defence_rating': 78, 'home_overall_rating': 80,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 5, 'away_conceded_goals_last_5': 10,
        'away_xg_last_5': 4.8, 'away_wins_last_5': 2,
        'away_draws_last_5': 1, 'away_losses_last_5': 2,
        'away_possession': 40, 'away_ga_per_90': 1.8,
        'away_performance_ga': 0.8, 'away_xg': 1.2,
        'away_xag': 1.4, 'away_prgc': 8, 'away_prgp': 12,
        'away_attack_rating': 72, 'away_midfield_rating': 71,
        'away_defence_rating': 68, 'away_overall_rating': 70,
        'away_days_since_last_match': 4
    },
    
    # A+ B+ C+ D-
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 10, 'home_conceded_goals_last_5': 5,
        'home_xg_last_5': 9.5, 'home_wins_last_5': 4,
        'home_draws_last_5': 1, 'home_losses_last_5': 0,
        'home_possession': 40, 'home_ga_per_90': 0.8,
        'home_performance_ga': 1.2, 'home_xg': 2.1,
        'home_xag': 2.3, 'home_prgc': 12, 'home_prgp': 8,
        'home_attack_rating': 82, 'home_midfield_rating': 81,
        'home_defence_rating': 78, 'home_overall_rating': 80,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 5, 'away_conceded_goals_last_5': 10,
        'away_xg_last_5': 4.8, 'away_wins_last_5': 2,
        'away_draws_last_5': 1, 'away_losses_last_5': 2,
        'away_possession': 60, 'away_ga_per_90': 1.8,
        'away_performance_ga': 0.8, 'away_xg': 1.2,
        'away_xag': 1.4, 'away_prgc': 8, 'away_prgp': 12,
        'away_attack_rating': 72, 'away_midfield_rating': 71,
        'away_defence_rating': 68, 'away_overall_rating': 70,
        'away_days_since_last_match': 4
    },
    
    # A+ B+ C- D+
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 10, 'home_conceded_goals_last_5': 5,
        'home_xg_last_5': 9.5, 'home_wins_last_5': 2,
        'home_draws_last_5': 1, 'home_losses_last_5': 2,
        'home_possession': 60, 'home_ga_per_90': 0.8,
        'home_performance_ga': 1.2, 'home_xg': 2.1,
        'home_xag': 2.3, 'home_prgc': 12, 'home_prgp': 8,
        'home_attack_rating': 82, 'home_midfield_rating': 81,
        'home_defence_rating': 78, 'home_overall_rating': 80,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 5, 'away_conceded_goals_last_5': 10,
        'away_xg_last_5': 4.8, 'away_wins_last_5': 4,
        'away_draws_last_5': 1, 'away_losses_last_5': 0,
        'away_possession': 40, 'away_ga_per_90': 1.8,
        'away_performance_ga': 0.8, 'away_xg': 1.2,
        'away_xag': 1.4, 'away_prgc': 8, 'away_prgp': 12,
        'away_attack_rating': 72, 'away_midfield_rating': 71,
        'away_defence_rating': 68, 'away_overall_rating': 70,
        'away_days_since_last_match': 4
    },
    
    # A+ B+ C- D-
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 10, 'home_conceded_goals_last_5': 5,
        'home_xg_last_5': 9.5, 'home_wins_last_5': 2,
        'home_draws_last_5': 1, 'home_losses_last_5': 2,
        'home_possession': 40, 'home_ga_per_90': 0.8,
        'home_performance_ga': 1.2, 'home_xg': 2.1,
        'home_xag': 2.3, 'home_prgc': 12, 'home_prgp': 8,
        'home_attack_rating': 82, 'home_midfield_rating': 81,
        'home_defence_rating': 78, 'home_overall_rating': 80,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 5, 'away_conceded_goals_last_5': 10,
        'away_xg_last_5': 4.8, 'away_wins_last_5': 4,
        'away_draws_last_5': 1, 'away_losses_last_5': 0,
        'away_possession': 60, 'away_ga_per_90': 1.8,
        'away_performance_ga': 0.8, 'away_xg': 1.2,
        'away_xag': 1.4, 'away_prgc': 8, 'away_prgp': 12,
        'away_attack_rating': 72, 'away_midfield_rating': 71,
        'away_defence_rating': 68, 'away_overall_rating': 70,
        'away_days_since_last_match': 4
    },
    
    # A+ B- C+ D+
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 5, 'home_conceded_goals_last_5': 10,
        'home_xg_last_5': 4.8, 'home_wins_last_5': 4,
        'home_draws_last_5': 1, 'home_losses_last_5': 0,
        'home_possession': 60, 'home_ga_per_90': 1.8,
        'home_performance_ga': 0.8, 'home_xg': 1.2,
        'home_xag': 1.4, 'home_prgc': 8, 'home_prgp': 12,
        'home_attack_rating': 72, 'home_midfield_rating': 71,
        'home_defence_rating': 68, 'home_overall_rating': 80,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 10, 'away_conceded_goals_last_5': 5,
        'away_xg_last_5': 9.5, 'away_wins_last_5': 2,
        'away_draws_last_5': 1, 'away_losses_last_5': 2,
        'away_possession': 40, 'away_ga_per_90': 0.8,
        'away_performance_ga': 1.2, 'away_xg': 2.1,
        'away_xag': 2.3, 'away_prgc': 12, 'away_prgp': 8,
        'away_attack_rating': 82, 'away_midfield_rating': 81,
        'away_defence_rating': 78, 'away_overall_rating': 70,
        'away_days_since_last_match': 4
    },
    
    # A+ B- C+ D-
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 5, 'home_conceded_goals_last_5': 10,
        'home_xg_last_5': 4.8, 'home_wins_last_5': 4,
        'home_draws_last_5': 1, 'home_losses_last_5': 0,
        'home_possession': 40, 'home_ga_per_90': 1.8,
        'home_performance_ga': 0.8, 'home_xg': 1.2,
        'home_xag': 1.4, 'home_prgc': 8, 'home_prgp': 12,
        'home_attack_rating': 72, 'home_midfield_rating': 71,
        'home_defence_rating': 68, 'home_overall_rating': 80,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 10, 'away_conceded_goals_last_5': 5,
        'away_xg_last_5': 9.5, 'away_wins_last_5': 2,
        'away_draws_last_5': 1, 'away_losses_last_5': 2,
        'away_possession': 60, 'away_ga_per_90': 0.8,
        'away_performance_ga': 1.2, 'away_xg': 2.1,
        'away_xag': 2.3, 'away_prgc': 12, 'away_prgp': 8,
        'away_attack_rating': 82, 'away_midfield_rating': 81,
        'away_defence_rating': 78, 'away_overall_rating': 70,
        'away_days_since_last_match': 4
    },
    
    # A+ B- C- D+
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 5, 'home_conceded_goals_last_5': 10,
        'home_xg_last_5': 4.8, 'home_wins_last_5': 2,
        'home_draws_last_5': 1, 'home_losses_last_5': 2,
        'home_possession': 60, 'home_ga_per_90': 1.8,
        'home_performance_ga': 0.8, 'home_xg': 1.2,
        'home_xag': 1.4, 'home_prgc': 8, 'home_prgp': 12,
        'home_attack_rating': 72, 'home_midfield_rating': 71,
        'home_defence_rating': 68, 'home_overall_rating': 80,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 10, 'away_conceded_goals_last_5': 5,
        'away_xg_last_5': 9.5, 'away_wins_last_5': 4,
        'away_draws_last_5': 1, 'away_losses_last_5': 0,
        'away_possession': 40, 'away_ga_per_90': 0.8,
        'away_performance_ga': 1.2, 'away_xg': 2.1,
        'away_xag': 2.3, 'away_prgc': 12, 'away_prgp': 8,
        'away_attack_rating': 82, 'away_midfield_rating': 81,
        'away_defence_rating': 78, 'away_overall_rating': 70,
        'away_days_since_last_match': 4
    },
    
    # A+ B- C- D-
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 5, 'home_conceded_goals_last_5': 10,
        'home_xg_last_5': 4.8, 'home_wins_last_5': 2,
        'home_draws_last_5': 1, 'home_losses_last_5': 2,
        'home_possession': 40, 'home_ga_per_90': 1.8,
        'home_performance_ga': 0.8, 'home_xg': 1.2,
        'home_xag': 1.4, 'home_prgc': 8, 'home_prgp': 12,
        'home_attack_rating': 72, 'home_midfield_rating': 71,
        'home_defence_rating': 68, 'home_overall_rating': 80,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 10, 'away_conceded_goals_last_5': 5,
        'away_xg_last_5': 9.5, 'away_wins_last_5': 4,
        'away_draws_last_5': 1, 'away_losses_last_5': 0,
        'away_possession': 60, 'away_ga_per_90': 0.8,
        'away_performance_ga': 1.2, 'away_xg': 2.1,
        'away_xag': 2.3, 'away_prgc': 12, 'away_prgp': 8,
        'away_attack_rating': 82, 'away_midfield_rating': 81,
        'away_defence_rating': 78, 'away_overall_rating': 70,
        'away_days_since_last_match': 4
    },
    
    # A- B+ C+ D+
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 10, 'home_conceded_goals_last_5': 5,
        'home_xg_last_5': 9.5, 'home_wins_last_5': 4,
        'home_draws_last_5': 1, 'home_losses_last_5': 0,
        'home_possession': 60, 'home_ga_per_90': 0.8,
        'home_performance_ga': 1.2, 'home_xg': 2.1,
        'home_xag': 2.3, 'home_prgc': 12, 'home_prgp': 8,
        'home_attack_rating': 72, 'home_midfield_rating': 71,
        'home_defence_rating': 68, 'home_overall_rating': 70,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 5, 'away_conceded_goals_last_5': 10,
        'away_xg_last_5': 4.8, 'away_wins_last_5': 2,
        'away_draws_last_5': 1, 'away_losses_last_5': 2,
        'away_possession': 40, 'away_ga_per_90': 1.8,
        'away_performance_ga': 0.8, 'away_xg': 1.2,
        'away_xag': 1.4, 'away_prgc': 8, 'away_prgp': 12,
        'away_attack_rating': 82, 'away_midfield_rating': 81,
        'away_defence_rating': 78, 'away_overall_rating': 80,
        'away_days_since_last_match': 4
    },
    
    # A- B+ C+ D-
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 10, 'home_conceded_goals_last_5': 5,
        'home_xg_last_5': 9.5, 'home_wins_last_5': 4,
        'home_draws_last_5': 1, 'home_losses_last_5': 0,
        'home_possession': 40, 'home_ga_per_90': 0.8,
        'home_performance_ga': 1.2, 'home_xg': 2.1,
        'home_xag': 2.3, 'home_prgc': 12, 'home_prgp': 8,
        'home_attack_rating': 72, 'home_midfield_rating': 71,
        'home_defence_rating': 68, 'home_overall_rating': 70,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 5, 'away_conceded_goals_last_5': 10,
        'away_xg_last_5': 4.8, 'away_wins_last_5': 2,
        'away_draws_last_5': 1, 'away_losses_last_5': 2,
        'away_possession': 60, 'away_ga_per_90': 1.8,
        'away_performance_ga': 0.8, 'away_xg': 1.2,
        'away_xag': 1.4, 'away_prgc': 8, 'away_prgp': 12,
        'away_attack_rating': 82, 'away_midfield_rating': 81,
        'away_defence_rating': 78, 'away_overall_rating': 80,
        'away_days_since_last_match': 4
    },
    
    # A- B+ C- D+
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 10, 'home_conceded_goals_last_5': 5,
        'home_xg_last_5': 9.5, 'home_wins_last_5': 2,
        'home_draws_last_5': 1, 'home_losses_last_5': 2,
        'home_possession': 60, 'home_ga_per_90': 0.8,
        'home_performance_ga': 1.2, 'home_xg': 2.1,
        'home_xag': 2.3, 'home_prgc': 12, 'home_prgp': 8,
        'home_attack_rating': 72, 'home_midfield_rating': 71,
        'home_defence_rating': 68, 'home_overall_rating': 70,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 5, 'away_conceded_goals_last_5': 10,
        'away_xg_last_5': 4.8, 'away_wins_last_5': 4,
        'away_draws_last_5': 1, 'away_losses_last_5': 0,
        'away_possession': 40, 'away_ga_per_90': 1.8,
        'away_performance_ga': 0.8, 'away_xg': 1.2,
        'away_xag': 1.4, 'away_prgc': 8, 'away_prgp': 12,
        'away_attack_rating': 82, 'away_midfield_rating': 81,
        'away_defence_rating': 78, 'away_overall_rating': 80,
        'away_days_since_last_match': 4
    },
    
    # A- B+ C- D-
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 10, 'home_conceded_goals_last_5': 5,
        'home_xg_last_5': 9.5, 'home_wins_last_5': 2,
        'home_draws_last_5': 1, 'home_losses_last_5': 2,
        'home_possession': 40, 'home_ga_per_90': 0.8,
        'home_performance_ga': 1.2, 'home_xg': 2.1,
        'home_xag': 2.3, 'home_prgc': 12, 'home_prgp': 8,
        'home_attack_rating': 72, 'home_midfield_rating': 71,
        'home_defence_rating': 68, 'home_overall_rating': 70,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 5, 'away_conceded_goals_last_5': 10,
        'away_xg_last_5': 4.8, 'away_wins_last_5': 4,
        'away_draws_last_5': 1, 'away_losses_last_5': 0,
        'away_possession': 60, 'away_ga_per_90': 1.8,
        'away_performance_ga': 0.8, 'away_xg': 1.2,
        'away_xag': 1.4, 'away_prgc': 8, 'away_prgp': 12,
        'away_attack_rating': 82, 'away_midfield_rating': 81,
        'away_defence_rating': 78, 'away_overall_rating': 80,
        'away_days_since_last_match': 4
    },
    
    # A- B- C+ D+
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 5, 'home_conceded_goals_last_5': 10,
        'home_xg_last_5': 4.8, 'home_wins_last_5': 4,
        'home_draws_last_5': 1, 'home_losses_last_5': 0,
        'home_possession': 60, 'home_ga_per_90': 1.8,
        'home_performance_ga': 0.8, 'home_xg': 1.2,
        'home_xag': 1.4, 'home_prgc': 8, 'home_prgp': 12,
        'home_attack_rating': 72, 'home_midfield_rating': 71,
        'home_defence_rating': 68, 'home_overall_rating': 70,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 10, 'away_conceded_goals_last_5': 5,
        'away_xg_last_5': 9.5, 'away_wins_last_5': 2,
        'away_draws_last_5': 1, 'away_losses_last_5': 2,
        'away_possession': 40, 'away_ga_per_90': 0.8,
        'away_performance_ga': 1.2, 'away_xg': 2.1,
        'away_xag': 2.3, 'away_prgc': 12, 'away_prgp': 8,
        'away_attack_rating': 82, 'away_midfield_rating': 81,
        'away_defence_rating': 78, 'away_overall_rating': 80,
        'away_days_since_last_match': 4
    },
    
    # A- B- C+ D-
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 5, 'home_conceded_goals_last_5': 10,
        'home_xg_last_5': 4.8, 'home_wins_last_5': 4,
        'home_draws_last_5': 1, 'home_losses_last_5': 0,
        'home_possession': 40, 'home_ga_per_90': 1.8,
        'home_performance_ga': 0.8, 'home_xg': 1.2,
        'home_xag': 1.4, 'home_prgc': 8, 'home_prgp': 12,
        'home_attack_rating': 72, 'home_midfield_rating': 71,
        'home_defence_rating': 68, 'home_overall_rating': 70,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 10, 'away_conceded_goals_last_5': 5,
        'away_xg_last_5': 9.5, 'away_wins_last_5': 2,
        'away_draws_last_5': 1, 'away_losses_last_5': 2,
        'away_possession': 60, 'away_ga_per_90': 0.8,
        'away_performance_ga': 1.2, 'away_xg': 2.1,
        'away_xag': 2.3, 'away_prgc': 12, 'away_prgp': 8,
        'away_attack_rating': 82, 'away_midfield_rating': 81,
        'away_defence_rating': 78, 'away_overall_rating': 80,
        'away_days_since_last_match': 4
    },
    
    # A- B- C- D+
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 5, 'home_conceded_goals_last_5': 10,
        'home_xg_last_5': 4.8, 'home_wins_last_5': 2,
        'home_draws_last_5': 1, 'home_losses_last_5': 2,
        'home_possession': 60, 'home_ga_per_90': 1.8,
        'home_performance_ga': 0.8, 'home_xg': 1.2,
        'home_xag': 1.4, 'home_prgc': 8, 'home_prgp': 12,
        'home_attack_rating': 72, 'home_midfield_rating': 71,
        'home_defence_rating': 68, 'home_overall_rating': 70,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 10, 'away_conceded_goals_last_5': 5,
        'away_xg_last_5': 9.5, 'away_wins_last_5': 4,
        'away_draws_last_5': 1, 'away_losses_last_5': 0,
        'away_possession': 40, 'away_ga_per_90': 0.8,
        'away_performance_ga': 1.2, 'away_xg': 2.1,
        'away_xag': 2.3, 'away_prgc': 12, 'away_prgp': 8,
        'away_attack_rating': 82, 'away_midfield_rating': 81,
        'away_defence_rating': 78, 'away_overall_rating': 80,
        'away_days_since_last_match': 4
    },
    
    # A- B- C- D-
    {
        'home_team': 'Chelsea', 'away_team': 'Roma',
        'home_goals_last_5': 5, 'home_conceded_goals_last_5': 10,
        'home_xg_last_5': 4.8, 'home_wins_last_5': 2,
        'home_draws_last_5': 1, 'home_losses_last_5': 2,
        'home_possession': 40, 'home_ga_per_90': 1.8,
        'home_performance_ga': 0.8, 'home_xg': 1.2,
        'home_xag': 1.4, 'home_prgc': 8, 'home_prgp': 12,
        'home_attack_rating': 72, 'home_midfield_rating': 71,
        'home_defence_rating': 68, 'home_overall_rating': 70,
        'home_days_since_last_match': 3,
        'away_goals_last_5': 10, 'away_conceded_goals_last_5': 5,
        'away_xg_last_5': 9.5, 'away_wins_last_5': 4,
        'away_draws_last_5': 1, 'away_losses_last_5': 0,
        'away_possession': 60, 'away_ga_per_90': 0.8,
        'away_performance_ga': 1.2, 'away_xg': 2.1,
        'away_xag': 2.3, 'away_prgc': 12, 'away_prgp': 8,
        'away_attack_rating': 82, 'away_midfield_rating': 81,
        'away_defence_rating': 78, 'away_overall_rating': 80,
        'away_days_since_last_match': 4
    }
]
    
    home_team = api.find_team_name("Newcastle")
    away_team = api.find_team_name("Bournemouth")
    match_data, _ = api.get_match_data(home_team, away_team, date)
    results_table = []
    for match_stats in matches:
        synth_match = match_data.copy()
        for field, value in match_stats.items():
            if field in synth_match.columns:
                synth_match[field] = value
        encoded_match_data = api.dp.encode_match_data(synth_match)
        results = api.predictor.predict_and_determine_winner(encoded_match_data)
        # Определяем комбинацию A,B,C,D (+, -)
        A = '+' if match_stats["home_overall_rating"] >= match_stats["away_overall_rating"] else '-'
        B = '+' if (match_stats["home_goals_last_5"] - match_stats["home_conceded_goals_last_5"]) >= \
                   (match_stats["away_goals_last_5"] - match_stats["away_conceded_goals_last_5"]) else '-'
        C = '+' if match_stats["home_wins_last_5"] >= match_stats["away_wins_last_5"] else '-'
        D = '+' if match_stats["home_possession"] > 50.0 else '-'
        
        # Добавляем результаты в таблицу
        results_table.append([
            A, B, C, D,
            results['combined_probabilities']['home_win'],
            results['combined_probabilities']['draw'],
            results['combined_probabilities']['away_win']
        ])
    results_df = pd.DataFrame(results_table, columns=[
        'A', 'B', 'C', 'D', 
        'P(Победа хозяев)', 'P(Ничья)', 'P(Победа гостей)'
    ])
    logger.debug("\nТаблица 5:")
    logger.debug(results_df.to_string(index=False, float_format="%.2f"))
    
    # Данные для конкретного матча
    home_team = api.find_team_name("Newcastle")
    away_team = api.find_team_name("Bournemouth")
    _, encoded_match_data = api.get_match_data(home_team, away_team, date)
    results_table = []
    results = api.predictor.predict_and_determine_winner(encoded_match_data)
    # Определяем комбинацию A,B,C,D (+, -)
    A = '+' if match_stats["home_overall_rating"] >= match_stats["away_overall_rating"] else '-'
    B = '+' if (match_stats["home_goals_last_5"] - match_stats["home_conceded_goals_last_5"]) >= \
                (match_stats["away_goals_last_5"] - match_stats["away_conceded_goals_last_5"]) else '-'
    C = '+' if match_stats["home_wins_last_5"] >= match_stats["away_wins_last_5"] else '-'
    D = '+' if match_stats["home_possession"] > 50.0 else '-'
    
    # Добавляем результаты в таблицу
    results_table.append([
        A, B, C, D,
        results['combined_probabilities']['home_win'],
        results['combined_probabilities']['draw'],
        results['combined_probabilities']['away_win']
    ])
    results_df = pd.DataFrame(results_table, columns=[
        'A', 'B', 'C', 'D', 
        'P(Победа хозяев)', 'P(Ничья)', 'P(Победа гостей)'
    ])
    logger.debug("\nТаблица для конкретного матча:")
    logger.debug(results_df.to_string(index=False, float_format="%.2f"))
    
    