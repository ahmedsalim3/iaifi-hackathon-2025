import pandas as pd
from tqdm import tqdm

import json
import os

# ===============================
# CONFIG
# ===============================
VOTES_CSV = "data/target/gz2_hart16.csv"
MAP_CSV = "data/target/gz2_filename_mapping.csv"
IMG_DIR = "data/target/images_gz2/images"
OUT_JSON_ALL = "data/target/labels_master.json"
OUT_JSON_TOP = "data/target/labels_master_top_n.json"

# --- THRESHOLDS ---
THRESHOLDS = {
    'artifact': 0.2,
    'elliptical': 0.95,
    'nospiral': 0.9,
    'spiral': 0.95,
    'features': 0.9,
    'edgeon_ell': 0.1,
    'edgeon_spiral': 0.2,
    'odd': 0.7,
    'irregular': 0.7
}

def load_data(votes_csv, map_csv):
    """
    Load and merge the Galaxy Zoo vote fractions and filename mapping.
    Returns a merged DataFrame with consistent 'objid' type.
    """
    votes_df = pd.read_csv(votes_csv, low_memory=False)
    map_df = pd.read_csv(map_csv, low_memory=False)

    if "dr7objid" in votes_df.columns:
        votes_df = votes_df.rename(columns={"dr7objid": "objid"})

    votes_df['objid'] = votes_df['objid'].astype(int)
    map_df['objid'] = map_df['objid'].astype(int)
    return votes_df.merge(map_df, on="objid", how="inner")


def classify_gz2(row):
    """
    Classify a galaxy into elliptical, spiral, or irregular
    using ultra-strict debiased vote fraction thresholds.
    Returns (label, metrics_dict).
    """
    m = {
        'artifact_prob': row['t01_smooth_or_features_a03_star_or_artifact_debiased'],
        'smooth_prob': row['t01_smooth_or_features_a01_smooth_debiased'],
        'features_prob': row['t01_smooth_or_features_a02_features_or_disk_debiased'],
        'edgeon_prob': row['t02_edgeon_a04_yes_debiased'],
        'spiral_prob': row['t04_spiral_a08_spiral_debiased'],
        'nospiral_prob': row['t04_spiral_a09_no_spiral_debiased'],
        'irregular_prob': row['t08_odd_feature_a22_irregular_debiased'],
        'merger_prob': row['t08_odd_feature_a24_merger_debiased'],
        'disturbed_prob': row['t08_odd_feature_a21_disturbed_debiased'],
        'odd_prob': row['t06_odd_a14_yes_debiased']
    }

    if m['artifact_prob'] >= THRESHOLDS['artifact']:
        return None, m

    if (m['smooth_prob'] >= THRESHOLDS['elliptical'] and
        m['edgeon_prob'] < THRESHOLDS['edgeon_ell'] and
        m['nospiral_prob'] >= THRESHOLDS['nospiral']):
        return "elliptical", m

    if (m['spiral_prob'] >= THRESHOLDS['spiral'] and
        m['features_prob'] >= THRESHOLDS['features'] and
        m['edgeon_prob'] < THRESHOLDS['edgeon_spiral']):
        return "spiral", m

    if (m['odd_prob'] >= THRESHOLDS['odd'] and
        max(m['irregular_prob'], m['merger_prob'], m['disturbed_prob']) >= THRESHOLDS['irregular']):
        return "irregular", m

    return None, m


def find_image_path(asset_id):
    """
    find the image path for a given asset_id
    Returns the path if found, else None
    """
    patterns = [
        f"{asset_id}.jpg",
        # f"{int(asset_id)}.jpg"
    ]
    for pat in patterns:
        p = os.path.join(IMG_DIR, pat)
        if os.path.exists(p):
            return p
    return None


def top_n(df, n_per_class):
    """
    For each class, select the top N galaxies sorted by the strongest
    confidence metric relevant to that class.
    """
    best_rows = []
    for cls in ['elliptical', 'spiral', 'irregular']:
        subset = df[df['classification'] == cls].copy()
        if cls == 'elliptical':
            subset['sort_value'] = subset['metrics'].apply(lambda x: x['smooth_prob'])
        elif cls == 'spiral':
            subset['sort_value'] = subset['metrics'].apply(lambda x: x['spiral_prob'])
        elif cls == 'irregular':
            subset['sort_value'] = subset['metrics'].apply(lambda x: max(
                x['irregular_prob'],
                x['merger_prob'],
                x['disturbed_prob']
            ))
        subset = subset.sort_values(by='sort_value', ascending=False)
        best_rows.append(subset.head(n_per_class))
    return pd.concat(best_rows)


def save_json(df, out_path):
    """
    Save the DataFrame to JSON with:
    - image path
    - objid
    - classification
    - metrics (vote fractions)
    """
    data = [
        {
            "image_path": row['image_path'],
            "objid": int(row['objid']),
            "classification": row['classification'],
            "metrics": row['metrics']
        }
        for _, row in df.iterrows()
    ]
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {len(data)} galaxies to {out_path}")


if __name__ == "__main__":
    df = load_data(VOTES_CSV, MAP_CSV)

    print("Classifying galaxies...")
    tqdm.pandas()
    df[['classification', 'metrics']] = df.progress_apply(
        lambda row: pd.Series(classify_gz2(row)), axis=1
    )

    print("Finding image paths...")
    df['image_path'] = df['asset_id'].progress_apply(find_image_path)

    valid_df = df[df['classification'].notnull() & df['image_path'].notnull()]

    print(valid_df['classification'].value_counts())

    # full clean dataset
    save_json(valid_df, OUT_JSON_ALL)

    # top-N per class dataset
    top_df = top_n(valid_df, 2000)
    print(top_df['classification'].value_counts())
    save_json(top_df, OUT_JSON_TOP)
