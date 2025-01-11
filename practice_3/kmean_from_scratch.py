import numpy as np

class My_KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        """
        Parameters:
            n_clusters: int, default=8
                The number of clusters to form.
            max_iter: int, default=300
                Maximum number of iterations of the k-means algorithm.
            tol: float, default=1e-4
                Tolerance to declare convergence.
            random_state: int, default=None
                Determines random number generation for centroid initialization.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        """
        Compute k-means clustering.
        
        Parameters:
            X: array-like, shape (n_samples, n_features)
                Training instances to cluster.
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        random_idxs = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[random_idxs]
        
        for _ in range(self.max_iter):
            # Assign labels based on closest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Compute new centroids
            new_centroids = []
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(axis=0))
                else:
                    # Reinitialize centroid for empty cluster
                    new_centroids.append(X[np.random.randint(0, n_samples)])
            
            new_centroids = np.array(new_centroids)
            
            # Check for convergence
            if np.all(np.abs(new_centroids - centroids) < self.tol):
                break
            
            centroids = new_centroids
    
        # Store results
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = np.sum((X - centroids[labels])**2)


    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters:
            X: array-like, shape (n_samples, n_features)
        
        Returns:
            labels: array, shape (n_samples,)
                Index of the cluster each sample belongs to.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)

