import numpy as np

def compute_stats(data):
    """Compute mean and variance of a given numpy array."""
    numpy_mean = np.mean(data)
    numpy_variance = np.var(data)

    manual_mean = sum(data) / len(data)
    manual_variance = sum((data - manual_mean) ** 2) / len(data)

    return numpy_mean, numpy_variance, manual_mean, manual_variance

def main():
    G = np.array([1.3, 1.7, 1.0, 2.0, 1.3, 1.7, 2.0, 2.3, 2.0, 1.7, 1.3, 1.0, 2.0, 1.7, 1.7, 1.3, 2.0])
    results = compute_stats(G)
    print("Mean and Variance using numpy's functions:", results[:2])
    print("Mean and Variance using manual implementation:", results[2:])

if __name__ == "__main__":
    main()
