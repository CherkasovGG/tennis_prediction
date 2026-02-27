import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config.config import models_dir


def load_all():
    cb_calib = joblib.load(os.path.join(models_dir, "cb_model.pkl"))
    rf_calib = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
    tabpfn_calib = joblib.load(os.path.join(models_dir, "tabpfn_model.pkl"))
    meta     = joblib.load(os.path.join(models_dir, "meta_model.pkl"))
    features = joblib.load(os.path.join(models_dir, "features_list.pkl"))
    dtypes   = joblib.load(os.path.join(models_dir, "train_dtypes.pkl"))
    enc      = joblib.load(os.path.join(models_dir, "rf_encoder.pkl"))
    last_test_df = 'popa'

    return cb_calib, rf_calib, tabpfn_calib, meta, features, dtypes, enc, last_test_df

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


def predict_match(player_a_name, player_b_name, odds_a, odds_b):
    cb_calib, rf_calib, tabpfn_calib, meta, features, dtypes, enc, last_test_df = load_all()

    mask_a_A = last_test_df['player_a'].astype(str).str.contains(player_a_name, case=False, na=False)
    mask_b_A = last_test_df['player_b'].astype(str).str.contains(player_a_name, case=False, na=False)

    row_a_source, as_side_a = None, None
    if mask_a_A.any():
        row_a_source = last_test_df[mask_a_A].iloc[-1]
        as_side_a = 'a'
    elif mask_b_A.any():
        row_a_source = last_test_df[mask_b_A].iloc[-1]
        as_side_a = 'b'

    mask_a_B = last_test_df['player_a'].astype(str).str.contains(player_b_name, case=False, na=False)
    mask_b_B = last_test_df['player_b'].astype(str).str.contains(player_b_name, case=False, na=False)

    row_b_source, as_side_b = None, None
    if mask_a_B.any():
        row_b_source = last_test_df[mask_a_B].iloc[-1]
        as_side_b = 'a'
    elif mask_b_B.any():
        row_b_source = last_test_df[mask_b_B].iloc[-1]
        as_side_b = 'b'

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
    
    scaler = StandardScaler()

    X_rf_live = enc.transform(one_match_df[features])
    X_tab_live = scaler.transform(X_rf_live)

    pred_cb = cb_calib.predict_proba(one_match_df[features])
    pred_rf = rf_calib.predict_proba(X_rf_live)
    pred_tab = tabpfn_calib.predict_proba(X_tab_live)

    live_meta = pd.DataFrame({
        "cb": cb_calib.predict_proba(one_match_df[features])[:, 1],
        "rf": rf_calib.predict_proba(X_rf_live)[:, 1],
        'tabpfn': tabpfn_calib.predict_proba(X_tab_live)[:, 1],
    })

    prob_a = meta.predict_proba(live_meta)[:, 1][0]
    prob_a = float(np.clip(prob_a, 0.01, 0.99))
    return prob_a
