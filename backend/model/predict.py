import os
import joblib
import numpy as np
import pandas as pd
from config import models_dir


BEST_MIN_EDGE       = 0.02603376255747001
BEST_KELLY_FRACTION = 0.996937068780426
BEST_MIN_ODDS       = 1.2229944317839916
BEST_MIN_CONF       = 0.700305905583701


class ValueBettingStrategy:
    def __init__(self, min_edge=0.03, kelly_fraction=0.25, min_odds=1.5, min_confidence=0.6):
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.min_odds = min_odds
        self.min_confidence = min_confidence

    def get_bookmaker_probs(self, odds_a, odds_b):
        if pd.isna(odds_a) or pd.isna(odds_b):
            return None, None
        if odds_a <= 1.0 or odds_b <= 1.0:
            return None, None
        p_a = 1.0 / odds_a
        p_b = 1.0 / odds_b
        margin = p_a + p_b
        return p_a / margin, p_b / margin

    def calculate_edge(self, model_prob, bookie_prob):
        return model_prob - bookie_prob

    def calculate_kelly(self, probability, odds):
        if odds <= 1.0 or probability <= 0 or probability >= 1.0:
            return 0.0
        kelly = (probability * odds - 1.0) / (odds - 1.0)
        return max(0.0, min(1.0, kelly))

    def decide(self, model_prob_a, odds_a, odds_b, bankroll=100.0):
        bookie_prob_a, bookie_prob_b = self.get_bookmaker_probs(odds_a, odds_b)
        if bookie_prob_a is None:
            return {
                'action': 'no_bet', 'stake': 0.0, 'edge': 0.0,
                'kelly': 0.0, 'reason': 'Invalid odds'
            }

        model_prob_b = 1.0 - model_prob_a
        edge_a = self.calculate_edge(model_prob_a, bookie_prob_a)
        edge_b = self.calculate_edge(model_prob_b, bookie_prob_b)

        edge, bet_side, odds, prob = None, None, None, None

        if model_prob_a >= self.min_confidence:
            edge = edge_a
            bet_side = 'a'
            odds = odds_a
            prob = model_prob_a

        if model_prob_b >= self.min_confidence:
            if edge is None or edge_b > edge:
                edge = edge_b
                bet_side = 'b'
                odds = odds_b
                prob = model_prob_b

        if bet_side is None:
            return {
                'action': 'no_bet', 'stake': 0.0, 'edge': 0.0,
                'kelly': 0.0,
                'reason': f'No side with model_prob >= {self.min_confidence:.3f}'
            }

        if edge < self.min_edge:
            return {
                'action': 'no_bet', 'stake': 0.0, 'edge': edge,
                'kelly': 0.0, 'reason': f'Edge {edge:.4f} < min {self.min_edge}'
            }

        if odds < self.min_odds:
            return {
                'action': 'no_bet', 'stake': 0.0, 'edge': edge,
                'kelly': 0.0, 'reason': f'Odds {odds:.3f} < min {self.min_odds}'
            }

        kelly = self.calculate_kelly(prob, odds)
        stake = bankroll * kelly * self.kelly_fraction

        return {
            'action': f'bet_{bet_side}',
            'stake': float(stake),
            'edge': float(edge),
            'kelly': float(kelly),
            'reason': f'Edge {edge:.2%}, Kelly {kelly:.2%}, Stake {stake:.2f}'
        }


def load_all():
    cb_calib = joblib.load(os.path.join(models_dir, "cb_model.pkl"))
    rf_calib = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
    meta     = joblib.load(os.path.join(models_dir, "meta_model.pkl"))
    features = joblib.load(os.path.join(models_dir, "features_list.pkl"))
    dtypes   = joblib.load(os.path.join(models_dir, "train_dtypes.pkl"))
    enc      = joblib.load(os.path.join(models_dir, "rf_encoder.pkl"))
    last_test_df = pd.read_csv(os.path.join(models_dir, 'df.csv'))
    return cb_calib, rf_calib, meta, features, dtypes, enc, last_test_df


def extract_player_features(row, features, prefix_player, as_player_side):
    data = {}
    if row is None:
        for f in features:
            if f.startswith(prefix_player):
                data[f] = 0
        return data

    side_prefix = f"player_{as_player_side}_"
    for f in features:
        if f.startswith(prefix_player):
            base = f.split(prefix_player, 1)[1]
            src_col = side_prefix + base
            data[f] = row[src_col] if src_col in row.index else 0
    return data

def find_player_row(last_test_df, raw_name):
    """
    raw_name может быть 'Fabian Marozsan' или 'Marozsan F.'.
    Ищем сначала по полному имени, потом по фамилии.
    """
    name = str(raw_name).strip()
    mask_a = last_test_df['player_a'].astype(str).str.contains(name, case=False, na=False)
    mask_b = last_test_df['player_b'].astype(str).str.contains(name, case=False, na=False)
    if mask_a.any() or mask_b.any():
        if mask_a.any():
            return last_test_df[mask_a].iloc[-1], 'a'
        else:
            return last_test_df[mask_b].iloc[-1], 'b'

    parts = name.replace('.', '').split()
    if len(parts) == 2:
        first, second = parts
        candidates = [first, second]
    else:
        candidates = [name]

    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        mask_a = last_test_df['player_a'].astype(str).str.contains(cand, case=False, na=False)
        mask_b = last_test_df['player_b'].astype(str).str.contains(cand, case=False, na=False)
        if mask_a.any() or mask_b.any():
            if mask_a.any():
                return last_test_df[mask_a].iloc[-1], 'a'
            else:
                return last_test_df[mask_b].iloc[-1], 'b'

    return None, None


def predict_match(player_a_name, player_b_name, odds_a, odds_b, bankroll=100.0):
    """
    Возвращает:
    - prob_a: вероятность победы A
    - decision: dict с action/stake/edge/kelly/reason от Kelly-стратегии (лучшие параметры)
    """
    cb_calib, rf_calib, meta, features, dtypes, enc, last_test_df = load_all()

    row_a_source, as_side_a = find_player_row(last_test_df, player_a_name)
    row_b_source, as_side_b = find_player_row(last_test_df, player_b_name)

    sample = {}

    src_for_tourney = row_a_source if row_a_source is not None else row_b_source
    if src_for_tourney is not None:
        sample['surface']       = src_for_tourney.get('surface', 'Clay')
        sample['draw_size']     = src_for_tourney.get('draw_size', 32)
        sample['tourney_level'] = src_for_tourney.get('tourney_level', 'A')
        sample['best_of']       = src_for_tourney.get('best_of', 3)
    else:
        sample['surface']       = 'Clay'
        sample['draw_size']     = 32
        sample['tourney_level'] = 'A'
        sample['best_of']       = 3

    if row_a_source is not None and as_side_a is not None:
        sample.update(extract_player_features(row_a_source, features, 'player_a_', as_side_a))
    else:
        for f in features:
            if f.startswith('player_a_'):
                sample[f] = 0

    if row_b_source is not None and as_side_b is not None:
        sample.update(extract_player_features(row_b_source, features, 'player_b_', as_side_b))
    else:
        for f in features:
            if f.startswith('player_b_'):
                sample[f] = 0

    if row_a_source is not None:
        for f in features:
            if f in row_a_source.index and f not in sample and not f.startswith('odds_'):
                sample[f] = row_a_source[f]
    else:
        for f in features:
            if f not in sample and not f.startswith('odds_'):
                sample[f] = 0

    sample['odds_b365_player_a'] = odds_a
    sample['odds_b365_player_b'] = odds_b
    sample['odds_ps_player_a']   = odds_a
    sample['odds_ps_player_b']   = odds_b
    sample['odds_max_player_a']  = odds_a
    sample['odds_max_player_b']  = odds_b
    sample['odds_avg_player_a']  = odds_a
    sample['odds_avg_player_b']  = odds_b

    def inv_prob(o):
        return 0 if o <= 1e-9 else 1.0 / o

    pa = inv_prob(odds_a)
    pb = inv_prob(odds_b)
    margin = pa + pb
    pa_n, pb_n = pa / margin, pb / margin

    sample['player_a_implied_prob_b365'] = pa_n
    sample['player_b_implied_prob_b365'] = pb_n
    sample['implied_prob_spread_b365']   = pa_n - pb_n

    sample['player_a_implied_prob_ps']   = pa_n
    sample['player_b_implied_prob_ps']   = pb_n
    sample['implied_prob_spread_ps']     = pa_n - pb_n

    sample['player_a_implied_prob_avg']  = pa_n
    sample['player_b_implied_prob_avg']  = pb_n
    sample['implied_prob_spread_avg']    = pa_n - pb_n

    sample['player_a_implied_prob_cb']   = pa_n
    sample['player_b_implied_prob_cb']   = pb_n
    sample['implied_prob_spread_cb']     = pa_n - pb_n

    for diff_col in [
        'elo_diff', 'surface_elo_diff',
        'form_5_diff', 'form_10_diff', 'form_20_diff',
        'win_streak_diff', 'lose_streak_diff',
        'surface_winrate_diff',
        'days_since_last_diff', 'matches_7d_diff', 'matches_30d_diff',
        'opponent_elo_diff',
    ]:
        if row_a_source is not None and diff_col in row_a_source.index:
            sample[diff_col] = row_a_source[diff_col]
        else:
            sample[diff_col] = 0

    sample['left_right_interaction'] = (
        row_a_source.get('left_right_interaction', 0) if row_a_source is not None else 0
    )

    one_match_df = pd.DataFrame([sample])[features]

    for col in features:
        if col not in one_match_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(dtypes[col]):
            one_match_df[col] = pd.to_numeric(one_match_df[col], errors='coerce')
        else:
            one_match_df[col] = one_match_df[col].astype(str)

    X_rf_live = enc.transform(one_match_df[features])

    live_meta = pd.DataFrame({
        "cb": cb_calib.predict_proba(one_match_df[features])[:, 1],
        "rf": rf_calib.predict_proba(X_rf_live)[:, 1],
    })

    prob_a = meta.predict_proba(live_meta)[:, 1][0]
    prob_a = float(np.clip(prob_a, 0.01, 0.99))

    strategy = ValueBettingStrategy(
        min_edge=BEST_MIN_EDGE,
        kelly_fraction=BEST_KELLY_FRACTION,
        min_odds=BEST_MIN_ODDS,
        min_confidence=BEST_MIN_CONF,
    )
    decision = strategy.decide(prob_a, odds_a, odds_b, bankroll=bankroll)

    return prob_a, decision