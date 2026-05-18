import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
FIG_DIR = BASE_DIR / "figures"
RESULT_DIR = BASE_DIR / "results"

FIG_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def read_csv_semicolon(filename):
    return pd.read_csv(RAW_DIR / filename, sep=";")


def plot_ahc_clusters(pg_class):
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        pg_class["longitude"],
        pg_class["latitude"],
        c=pg_class["AHC_class"],
        alpha=0.8
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Spatial livestock monitoring points by AHC class")
    plt.colorbar(scatter, label="AHC class")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pg_ahc_spatial_clusters.png", dpi=300)
    plt.close()


def plot_class_counts(pg_class):
    class_counts = pg_class["AHC_class"].value_counts().sort_index()
    class_counts.to_csv(RESULT_DIR / "pg_ahc_class_counts.csv")

    plt.figure(figsize=(7, 4))
    class_counts.plot(kind="bar")
    plt.xlabel("AHC class")
    plt.ylabel("Number of monitoring points")
    plt.title("Number of livestock monitoring points per AHC class")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pg_ahc_class_counts.png", dpi=300)
    plt.close()


def analyze_slope_by_class(pg_class, pg_slope):
    """
    Assign the nearest slope category to each livestock monitoring point
    and analyze slope distribution by AHC class.
    """

    class_points = pg_class[["latitude", "longitude"]].copy()
    slope_points = pg_slope[["latitude", "longitude"]].copy()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(slope_points)

    distances, indices = nn.kneighbors(class_points)

    merged = pg_class.copy()
    merged["nearest_slope"] = pg_slope.iloc[indices.flatten()]["slope"].values
    merged["distance_to_slope_point"] = distances.flatten()

    merged.to_csv(RESULT_DIR / "pg_ahc_with_nearest_slope.csv", index=False)

    slope_table = pd.crosstab(
        merged["AHC_class"],
        merged["nearest_slope"],
        normalize="index"
    ) * 100

    slope_table.to_csv(RESULT_DIR / "pg_slope_distribution_by_ahc_class.csv")

    plt.figure(figsize=(8, 5))
    slope_table.plot(kind="bar", stacked=True, ax=plt.gca())
    plt.xlabel("AHC class")
    plt.ylabel("Percentage of points (%)")
    plt.title("Nearest slope distribution within each AHC class")
    plt.legend(title="Slope", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pg_slope_distribution_by_ahc_class.png", dpi=300)
    plt.close()


def run_kmeans_spatial_clustering(pg_class):
    features = pg_class[["latitude", "longitude"]]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
    pg_class["KMeans_cluster"] = kmeans.fit_predict(features_scaled) + 1

    pg_class.to_csv(RESULT_DIR / "pg_ahc_with_kmeans_clusters.csv", index=False)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        pg_class["longitude"],
        pg_class["latitude"],
        c=pg_class["KMeans_cluster"],
        alpha=0.8
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("KMeans clustering of livestock spatial monitoring points")
    plt.colorbar(scatter, label="KMeans cluster")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pg_kmeans_spatial_clusters.png", dpi=300)
    plt.close()


def main():
    pg_class = read_csv_semicolon("PG_AHC_class.csv")
    pg_slope = read_csv_semicolon("PG_slope.csv")

    plot_ahc_clusters(pg_class)
    plot_class_counts(pg_class)
    analyze_slope_by_class(pg_class, pg_slope)
    run_kmeans_spatial_clustering(pg_class)

    print("Analysis completed.")
    print(f"Figures saved in: {FIG_DIR}")
    print(f"Results saved in: {RESULT_DIR}")


if __name__ == "__main__":
    main()