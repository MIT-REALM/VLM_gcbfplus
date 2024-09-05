
import jax.numpy as jnp
from jax.lax import fori_loop
from jax import random
from jax.nn import one_hot
def kmeans(states, k, num_iters=100, tol=1e-4):
    """
    Perform k-means clustering using JAX.

    Parameters:
    states (jax.numpy.ndarray): Input data points of shape (N, 2).
    k (int): Number of clusters.
    num_iters (int): Maximum number of iterations.
    tol (float): Tolerance for convergence.

    Returns:
    jax.numpy.ndarray: Cluster centers of shape (k, 2).
    jax.numpy.ndarray: Cluster assignments of shape (N,).
    """
    N, D = states.shape
    
    
    key = random.PRNGKey(0)
    indices = random.choice(key, N, shape=(k,), replace=False)
    centroids = states[indices]

    def assign_clusters(states, centroids):
        """Assign each point to the nearest centroid."""
        distances = jnp.sqrt(jnp.sum((states[:, None, :] - centroids[None, :, :])**2, axis=-1))
        return jnp.argmin(distances, axis=1)

    def update_centroids(states, assignments, k):
        """Update centroids as the mean of assigned points."""
        one_hot_assignments = one_hot(assignments, k)
        sum_states = jnp.dot(one_hot_assignments.T, states)
        count_states = one_hot_assignments.sum(axis=0)[:, None]
        new_centroids = sum_states / jnp.maximum(count_states, 1)  
        return new_centroids

    def step(i, state):
        centroids, old_centroids, assignments = state
        assignments = assign_clusters(states, centroids)
        new_centroids = update_centroids(states, assignments, k)
        return new_centroids, centroids, assignments

    centroids, old_centroids, assignments = fori_loop(0, num_iters, step, (centroids, centroids + 2 * tol, jnp.zeros(N, dtype=jnp.int32)))
    
    return centroids, assignments

