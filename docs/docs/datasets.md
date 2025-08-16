Datasets
========

This project uses two main datasets for domain adaptation research in galaxy morphology classification.
## Overview

We work with a **source dataset** from simulated galaxies and a **target dataset** from real galaxy observations to study domain adaptation techniques.

## Source Dataset (IllustrisTNG)

The source dataset is based on the [IllustrisTNG][tng-website] Project, with morphology labels available [here][tng-labels]. It contains three morphology classes: elliptical, spiral, and irregular galaxies.

### Data Structure

Each entry in the source dataset contains the following information:

```json
{
  "image_path": "local/path/to/image/subhalo_{subhalo_id}.png",
  "subhalo_id": "Unique galaxy ID matching image filename",
  "mass": "Total stellar mass [in solar masses, M_sun]",
  "star_forming": "true if galaxy is actively forming stars, false otherwise",
  "has_agn": "true if galaxy hosts an active galactic nucleus (AGN), false otherwise",
  "is_compact": "true if galaxy is compact (stellar mass / half-mass-radius^2 > 1e9 M_sun/kpc^2), false otherwise",
  "metallicity": "Stellar metallicity [in units of solar metallicity Z]",
  "classification": "'elliptical', 'spiral', or 'irregular'",
  "is_metalrich": "true if average stellar metallicity > solar, false otherwise"
}
```

## Target Dataset (Galaxy Zoo 2)

The target dataset is constructed from the [Galaxy Zoo 2 (GZ2)][gz2hart] project, using the debiased vote fractions provided by [Hart et al. (2016)][gz2hart-paper].

### Methodology

Hart et al. presented a method to correct classification bias in GZ2 by providing *debiased vote fractions* for morphological features (e.g., smooth, features/disk, spiral arms, odd structures). These corrected vote fractions represent the probability that a galaxy belongs to a given morphological type after accounting for redshift and observational effects. In the paper, thresholds on these debiased votes can be applied to extract clean samples of galaxies with reliable morphology (e.g., high-confidence ellipticals or spirals).

### Data Processing

We followed the paper's approach of applying strict thresholds to these debiased vote fractions in order to obtain an *ultra-clean target dataset* containing only the most confidently labeled galaxies. Specifically:

- **Ellipticals** were required to have very high smoothness probability, strong "no spiral" votes, and low edge-on contamination
- **Spirals** were required to have strong spiral-arm and feature/disk votes, while excluding highly edge-on systems
- **Irregulars** were selected when galaxies had strong votes for "odd/irregular," "merger," or "disturbed" features
- Galaxies flagged as *artifacts* (stars, imaging issues) were removed

### Data Structure

The resulting dataset is stored in JSON format with image paths, object IDs, classification labels, and confidence metrics used for filtering. Each entry contains the following information:

```json
{
  "image_path": "local/path/to/image/{asset_id}.png",
  "objid": "SDSS Data Release 7 object ID for cross-referencing with astronomical catalogs",
  "classification": "'elliptical', 'spiral', or 'irregular'",
  "metrics": {
    "artifact_prob": "Probability of being an artifact (star or imaging issue) - galaxies with high values are filtered out",
    "smooth_prob": "Probability of having smooth, featureless morphology (high values indicate elliptical galaxies)",
    "features_prob": "Probability of having disk features or structure (high values indicate spiral/disk galaxies)",
    "edgeon_prob": "Probability of being viewed edge-on (high values may indicate classification uncertainty)",
    "spiral_prob": "Probability of having spiral arm structure (high values indicate spiral galaxies)",
    "nospiral_prob": "Probability of having no spiral structure (high values support elliptical classification)",
    "irregular_prob": "Probability of having irregular, asymmetric morphology",
    "merger_prob": "Probability of being involved in a galaxy merger",
    "disturbed_prob": "Probability of having disturbed or unusual structure",
    "odd_prob": "Probability of having any odd or unusual features"
  }
}
```

The resulting dataset can be found on [GitHub][gz2-labels-topn]. In order to match the source dataset of 1,500 galaxies, we select the top N galaxies for each class, sorted by the strongest confidence metric relevant to that class.

### Data Processing Implementation

The preprocessing script that produced this target dataset implementation can be found in [`nebula/data/gz2.py`][gz2-preprocessing].

## Data Access

### Dataset Labels

- **Source Dataset (IllustrisTNG)**: [`data/source/llustrisTNG/labels_master.json`][tng-labels]
- **Target Dataset (GZ2)**: 
  - Full dataset: [`data/target/labels_master.json`][gz2-labels]
  - Top N subset: [`data/target/labels_master_top_n.json`][gz2-labels-topn]

### Image Downloads

You can download galaxy images using the [Python script][download-py] or [bash scripts][data-scripts].

### Data Directory Structure

After downloading, your data directory will be organized as follows:

```
data/
├── source/
│   └── llustrisTNG/
│       ├── images/           # Galaxy images
│       └── labels_master.json
└── target/
    ├── images_gz2/          # Galaxy images
    ├── labels_master.json
    ├── labels_master_top_n.json
    └── README.txt
```

[tng-website]: https://www.tng-project.org/
[gz2hart]: https://data.galaxyzoo.org/#section-8
[gz2hart-paper]: https://academic.oup.com/mnras/article/461/4/3663/2608720?login=false
[tng-labels]: https://github.com/ahmedsalim3/domain-adaptation-in-galaxy-morphology/blob/main/data/source/llustrisTNG/labels_master.json
[gz2-labels-topn]: https://github.com/ahmedsalim3/domain-adaptation-in-galaxy-morphology/blob/main/data/target/labels_master_top_n.json
[gz2-labels]: https://github.com/ahmedsalim3/domain-adaptation-in-galaxy-morphology/blob/main/data/target/labels_master.json
[gz2-preprocessing]: https://github.com/ahmedsalim3/domain-adaptation-in-galaxy-morphology/blob/main/nebula/data/gz2.py
[download-py]: https://github.com/ahmedsalim3/domain-adaptation-in-galaxy-morphology/blob/main/data/download_raw.py
[data-scripts]: https://github.com/ahmedsalim3/domain-adaptation-in-galaxy-morphology/blob/main/scripts/data