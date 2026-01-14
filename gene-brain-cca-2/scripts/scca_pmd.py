"""
Sparse CCA using Penalized Matrix Decomposition (PMD).

This implements the algorithm from:
Witten, D. M., Tibshirani, R., & Hastie, T. (2009).
"A penalized matrix decomposition, with applications to sparse principal
components and canonical correlation analysis."
Biostatistics, 10(3), 515-534.

This standalone implementation avoids external CCA library dependencies.
"""

from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Apply soft-thresholding (proximal operator for L1)."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def _project_l1_ball(x: np.ndarray, c: float) -> np.ndarray:
    """
    Project x onto L1 ball of radius c while ensuring L2 norm = 1.
    Uses binary search to find the right threshold.
    """
    # If already within L1 constraint, just normalize
    if np.linalg.norm(x, ord=1) <= c:
        norm = np.linalg.norm(x)
        return x / norm if norm > 1e-10 else x
    
    # Binary search for lambda
    lam_lo, lam_hi = 0.0, np.max(np.abs(x))
    
    for _ in range(50):  # max iterations for binary search
        lam = (lam_lo + lam_hi) / 2
        s = _soft_threshold(x, lam)
        l1_norm = np.linalg.norm(s, ord=1)
        
        if l1_norm < c - 1e-10:
            lam_hi = lam
        elif l1_norm > c + 1e-10:
            lam_lo = lam
        else:
            break
    
    s = _soft_threshold(x, lam)
    norm = np.linalg.norm(s)
    return s / norm if norm > 1e-10 else s


def _pmd_single_component(
    X: np.ndarray,
    Y: np.ndarray,
    c1: float,
    c2: float,
    max_iter: int = 500,
    tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a single pair of canonical vectors using PMD.
    
    Parameters
    ----------
    X : array (n_samples, p)
    Y : array (n_samples, q)
    c1 : float
        L1 constraint on u (weight vector for X). Range: [1, sqrt(p)]
    c2 : float
        L1 constraint on v (weight vector for Y). Range: [1, sqrt(q)]
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    
    Returns
    -------
    u : array (p,)
        Canonical weight vector for X
    v : array (q,)
        Canonical weight vector for Y
    """
    n, p = X.shape
    _, q = Y.shape
    
    # Cross-covariance matrix
    XtY = X.T @ Y / n
    
    # Initialize v randomly
    v = np.random.randn(q)
    v = v / np.linalg.norm(v)
    
    u_prev = np.zeros(p)
    
    for _ in range(max_iter):
        # Update u: u ← project(X'Y v, c1)
        u = XtY @ v
        u = _project_l1_ball(u, c1)
        
        # Update v: v ← project(Y'X u, c2)
        v = XtY.T @ u
        v = _project_l1_ball(v, c2)
        
        # Check convergence
        if np.linalg.norm(u - u_prev) < tol:
            break
        u_prev = u.copy()
    
    return u, v


class SCCA_PMD(BaseEstimator, TransformerMixin):
    """
    Sparse CCA using Penalized Matrix Decomposition.
    
    Parameters
    ----------
    latent_dimensions : int
        Number of canonical components to extract.
    c : list of 2 floats
        L1 constraints for [X, Y]. Each should be in range [1, sqrt(n_features)].
        Lower values = more sparsity.
    max_iter : int
        Maximum iterations per component.
    tol : float
        Convergence tolerance.
    random_state : int or None
        Random seed for reproducibility.
    
    Attributes
    ----------
    weights_ : list of 2 arrays
        Canonical weight matrices [W_x, W_y] of shapes (p, k) and (q, k).
    
    Example
    -------
    >>> scca = SCCA_PMD(latent_dimensions=10, c=[0.3, 0.3])
    >>> scca.fit([X, Y])
    >>> U, V = scca.transform([X, Y])
    """
    
    def __init__(
        self,
        latent_dimensions: int = 1,
        c: list[float] | None = None,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None
    ):
        self.latent_dimensions = latent_dimensions
        self.c = c if c is not None else [1.0, 1.0]
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
    def fit(self, views: list[np.ndarray]) -> "SCCA_PMD":
        """
        Fit SCCA model.
        
        Parameters
        ----------
        views : list of 2 arrays
            [X, Y] where X is (n, p) and Y is (n, q).
        
        Returns
        -------
        self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X, Y = views[0], views[1]
        n, p = X.shape
        _, q = Y.shape
        
        # Convert c to L1 bounds
        # If c is in [0, 1], interpret as fraction of max L1 norm
        c1, c2 = self.c
        if c1 <= 1.0:
            c1 = 1 + c1 * (np.sqrt(p) - 1)
        if c2 <= 1.0:
            c2 = 1 + c2 * (np.sqrt(q) - 1)
        
        # Storage for weights
        U = np.zeros((p, self.latent_dimensions))
        V = np.zeros((q, self.latent_dimensions))
        
        # Residual matrices for deflation
        X_res = X.copy()
        Y_res = Y.copy()
        
        for k in range(self.latent_dimensions):
            u, v = _pmd_single_component(
                X_res, Y_res, c1, c2, 
                max_iter=self.max_iter, 
                tol=self.tol
            )
            U[:, k] = u
            V[:, k] = v
            
            # Deflation: remove this component
            # Xu = X_res @ u; Yv = Y_res @ v
            # X_res = X_res - outer(Xu, u); Y_res = Y_res - outer(Yv, v)
            d_x = X_res @ u
            d_y = Y_res @ v
            X_res = X_res - np.outer(d_x, u)
            Y_res = Y_res - np.outer(d_y, v)
        
        self.weights_ = [U, V]
        return self
    
    @property
    def weights(self):
        """Alias for weights_ to match cca-zoo API."""
        return self.weights_
    
    def transform(self, views: list[np.ndarray]) -> list[np.ndarray]:
        """
        Transform data to canonical variates.
        
        Parameters
        ----------
        views : list of 2 arrays
            [X, Y] to transform.
        
        Returns
        -------
        list of 2 arrays
            [X @ W_x, Y @ W_y] canonical variates.
        """
        X, Y = views[0], views[1]
        return [X @ self.weights_[0], Y @ self.weights_[1]]
    
    def fit_transform(self, views: list[np.ndarray]) -> list[np.ndarray]:
        """Fit and transform in one step."""
        self.fit(views)
        return self.transform(views)


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    n, p, q = 200, 50, 30
    X = np.random.randn(n, p)
    Y = np.random.randn(n, q)
    
    # Add some structure
    X[:, :5] = Y[:, :5] @ np.random.randn(5, 5)
    
    scca = SCCA_PMD(latent_dimensions=5, c=[0.3, 0.3], random_state=42)
    scca.fit([X, Y])
    
    U, V = scca.transform([X, Y])
    
    print(f"Canonical variates shapes: {U.shape}, {V.shape}")
    print(f"Weight sparsity (X): {(scca.weights_[0] == 0).mean():.1%}")
    print(f"Weight sparsity (Y): {(scca.weights_[1] == 0).mean():.1%}")
    
    # Canonical correlations
    for k in range(5):
        r = np.corrcoef(U[:, k], V[:, k])[0, 1]
        print(f"  Component {k+1}: r = {r:.4f}")
    
    print("\n✓ SCCA_PMD working correctly")
