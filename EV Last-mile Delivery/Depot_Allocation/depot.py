import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import concurrent.futures

def chunk_coordinates(points, batch_size):
    """Splits the given points into batches."""
    for i in range(0, len(points), batch_size):
        yield points[i:i + batch_size]

def get_osrm_distances_osrm_table(depot, points):
    """
    Calculates the distances (km) from the depot to all demand points using OSRM.
    depot: [lon, lat]
    points: array of [lon, lat]
    """
    coords = [depot] + points.tolist()
    coord_str = ";".join([f"{lon},{lat}" for lon, lat in coords])

    url = f"http://localhost:5000/table/v1/driving/{coord_str}?sources=0&annotations=distance"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        distances = np.array(data['distances'][0][1:]) / 1000  # meters → kilometers
        return distances
    else:
        print("OSRM error:", response.text)
        return np.full(len(points), np.inf)

def get_osrm_batch_distances(batch, depot):
    """Gets the distances for a single batch using OSRM."""
    return get_osrm_distances_osrm_table(depot, batch)

def get_osrm_distances_parallel(depot, points, batch_size=500):
    """Gets distances from OSRM in parallel."""
    batches = list(chunk_coordinates(points, batch_size))

    distances = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda batch: get_osrm_batch_distances(batch, depot), batches)
        for result in results:
            distances.extend(result)

    return np.array(distances)

def find_weighted_centroid_depot_osrm(points, weights, bbox, step=0.05):
    """Uses grid search to find the weighted centroid depot location using OSRM distances."""
    lons = np.arange(bbox[0], bbox[2], step)
    lats = np.arange(bbox[1], bbox[3], step)

    best_point = None
    best_cost = float('inf')

    for lon in lons:
        for lat in lats:
            depot = [lon, lat]
            try:
                distances = get_osrm_distances_parallel(depot, points)
                weighted_sum = np.sum(weights * distances)

                if weighted_sum < best_cost:
                    best_cost = weighted_sum
                    best_point = (lon, lat)
            except Exception as e:
                print(f"Error at ({lon}, {lat}): {e}")
                continue

    return np.array([best_point]), best_cost

def plot_results(points, depots, bbox, title="Demand Points and Depot"):
    """Basic plot showing demand points and depot locations."""
    plt.figure(figsize=(10, 8))
    plt.xlim(bbox[0], bbox[2])
    plt.ylim(bbox[1], bbox[3])

    plt.scatter(points[:, 0], points[:, 1], c='blue', s=5, alpha=0.5, label='Demand Points')
    plt.scatter(depots[:, 0], depots[:, 1], c='red', s=80, marker='x', linewidths=2, label='Depot')

    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print("=== Starting OSRM-Based Weighted Centroid Analysis ===")

    # Bounding box (adjust if needed)
    bbox = (4.46, 50.71, 5.19, 51.05)  # Flanders / Belgium region

    # Load demand points and weights
    try:
        df = pd.read_excel("C:/Users/30572/Desktop/Combined_Data.xlsx", sheet_name="Merge3")
        points = df[['Long', 'Lat']].values
        weights = df['Location_Weight'].values
        print(f"Loaded {len(points)} demand points.")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # Run weighted centroid optimization
    print("\n=== Calculating Weighted Centroid Depot Location ===")
    depot, cost = find_weighted_centroid_depot_osrm(points, weights, bbox, step=0.05)
    print(f"Depot Location: Longitude = {depot[0][0]:.6f}, Latitude = {depot[0][1]:.6f}")
    print(f"Total Weighted OSRM Distance: {cost:.2f} km")

    # Plot results
    plot_results(points, depot, bbox, title="Weighted Centroid Depot Location (OSRM)")

    print("=== Analysis Complete ===")

if __name__ == "__main__":
    main()
