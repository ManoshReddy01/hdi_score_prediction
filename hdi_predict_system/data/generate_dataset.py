"""
Builds data/hdi_dataset.csv.

Ground truth HDI is computed with the actual UNDP methodology:
  LEI  = (life_expectancy - 20) / (85 - 20)
  MYSI = mean_years_schooling / 15
  EYSI = expected_years_schooling / 18
  EI   = (MYSI + EYSI) / 2
  II   = (ln(gni) - ln(100)) / (ln(75000) - ln(100))
  HDI  = (LEI * EI * II) ** (1/3)

Tiers (official UNDP bands):
  Very High >= 0.800   High 0.700-0.799   Medium 0.550-0.699   Low < 0.550
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from countries import COUNTRIES  # noqa: E402

RNG = np.random.default_rng(42)

TIER_RANGES = {
    "very_high": dict(le=(78, 84), mys=(11.5, 14.0), eys=(15.5, 18.5), gni=(38000, 85000)),
    "high": dict(le=(72, 78), mys=(8.5, 11.5), eys=(13.0, 16.0), gni=(12000, 32000)),
    "medium": dict(le=(64, 73), mys=(5.0, 8.5), eys=(9.5, 13.5), gni=(3000, 13000)),
    "low": dict(le=(50, 64), mys=(1.5, 5.0), eys=(5.0, 9.5), gni=(500, 3200)),
}


def hdi_score(le, mys, eys, gni):
    lei = (le - 20) / (85 - 20)
    mysi = mys / 15
    eysi = eys / 18
    ei = (mysi + eysi) / 2
    ii = (math.log(gni) - math.log(100)) / (math.log(75000) - math.log(100))
    lei, ei, ii = (max(0.001, min(1, v)) for v in (lei, ei, ii))
    return round((lei * ei * ii) ** (1 / 3), 4)


def category(score):
    if score >= 0.800:
        return "Very High"
    if score >= 0.700:
        return "High"
    if score >= 0.550:
        return "Medium"
    return "Low"


def build(out_path):
    rows = []
    for country, region, population, tier in COUNTRIES:
        r = TIER_RANGES[tier]
        le = float(np.round(RNG.uniform(*r["le"]), 1))
        mys = float(np.round(RNG.uniform(*r["mys"]), 1))
        eys = float(np.round(RNG.uniform(*r["eys"]), 1))
        gni = float(np.round(RNG.uniform(*r["gni"]), 0))

        score = hdi_score(le, mys, eys, gni)
        rows.append(
            {
                "country_name": country,
                "region": region,
                "population": population,
                "life_expectancy": le,
                "mean_years_schooling": mys,
                "expected_years_schooling": eys,
                "gni_per_capita": gni,
                "hdi_score": score,
                "hdi_category": category(score),
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "hdi_dataset.csv")
    df = build(path)
    print(f"Wrote {len(df)} rows to {path}")
    print(df["hdi_category"].value_counts())
