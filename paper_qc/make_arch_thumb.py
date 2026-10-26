"""Thumbnail of the study box for the detailed method figure (manuscript/method_detail.tex).

A bare map - land, and every profile coloured by split group as in figures/map.pdf - with no
axes or legend, small enough to sit in a card of the figure.

    python -m paper_qc.make_arch_thumb      ->  figures/thumb_map.png
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from paper_qc import profiles

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "thumb_map.png")
GROUPS = [("train", "#b9b8b0", 0.35), ("val", "#1baf7a", 0.6), ("test_clean", "#2a78d6", 0.6),
          ("test_known_bad", "#eb6834", 0.95)]


def main():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    p = pd.read_parquet(profiles.OUT, columns=["PLATFORM", "CYCLE", "LATITUDE", "LONGITUDE", "group"])
    p = p.groupby(["PLATFORM", "CYCLE"]).first()
    fig = plt.figure(figsize=(1.5, 1.35))
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent([20, 120, -60, 30])
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#e6e4dc", edgecolor="none")
    for g, color, alpha in GROUPS:
        q = p[p.group == g]
        ax.scatter(q.LONGITUDE, q.LATITUDE, s=0.25, color=color, alpha=alpha, linewidths=0,
                   transform=ccrs.PlateCarree(), rasterized=True)
    ax.spines["geo"].set_visible(False)
    fig.savefig(OUT, dpi=400)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
