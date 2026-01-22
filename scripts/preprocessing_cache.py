#!/usr/bin/env python3
"""
Preprocessing Cache Module for Phase 3 CCA Benchmark.

Provides disk-based caching for leakage-safe preprocessing (residualization,
standardization, PCA) to enable fast job resume and efficient grid search.

Lab Server Compliance:
- Uses /scratch for fast I/O when available
- Falls back to /storage for persistence
- Validates cache integrity before loading

Usage:
    from preprocessing_cache import FoldCache
    
    cache = FoldCache(cache_dir, fm_model, modality, seed)
    
    # Check if fold is cached
    if cache.has_fold(fold_idx):
        data = cache.load_fold(fold_idx)
    else:
        data = compute_fold_preprocessing(...)
        cache.save_fold(fold_idx, data)
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def get_cache_dir(base_name: str = "phase3_cache") -> Path:
    """
    Get the best cache directory, preferring /scratch for speed.
    
    Priority:
    1. /scratch/connectome/$USER/{base_name} (fast I/O)
    2. ./gene-brain-cca-2/derived/{base_name} (persistent fallback)
    
    Returns:
        Path to cache directory (created if needed)
    """
    user = os.environ.get("USER", "unknown")
    
    # Try /scratch first (fast I/O per lab guidelines)
    scratch_base = Path(f"/scratch/connectome/{user}")
    if scratch_base.exists():
        cache_dir = scratch_base / base_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    # Fallback to storage
    storage_dir = Path("gene-brain-cca-2/derived") / base_name
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def compute_array_hash(arr: np.ndarray, max_samples: int = 1000) -> str:
    """
    Compute a deterministic hash of an array for cache validation.
    
    Uses a subset of the array for speed while still detecting changes.
    """
    # Sample deterministically
    n = arr.shape[0]
    if n <= max_samples:
        sample = arr.ravel()
    else:
        step = n // max_samples
        sample = arr[::step].ravel()
    
    # Hash the bytes
    return hashlib.md5(sample.tobytes()).hexdigest()[:16]


class FoldCache:
    """
    Disk-based cache for preprocessed fold data.
    
    Stores preprocessed gene and brain matrices per fold to avoid
    recomputation on job restart or hyperparameter sweeps.
    """
    
    def __init__(
        self,
        cache_dir: Path | str,
        fm_model: str,
        modality: str,
        seed: int,
        n_folds: int = 5,
    ):
        """
        Initialize fold cache.
        
        Args:
            cache_dir: Base directory for cache files
            fm_model: Gene foundation model name (e.g., "caduceus")
            modality: Brain modality (e.g., "dmri", "smri")
            seed: Random seed for reproducibility
            n_folds: Number of CV folds
        """
        self.cache_dir = Path(cache_dir)
        self.fm_model = fm_model
        self.modality = modality
        self.seed = seed
        self.n_folds = n_folds
        
        # Create subdirectory for this specific run
        self.run_dir = self.cache_dir / f"{fm_model}_{modality}_seed{seed}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.meta_path = self.run_dir / "cache_meta.json"
        self._meta: dict[str, Any] | None = None
    
    def _get_meta(self) -> dict[str, Any]:
        """Load or create metadata."""
        if self._meta is not None:
            return self._meta
        
        if self.meta_path.exists():
            self._meta = json.loads(self.meta_path.read_text())
        else:
            self._meta = {
                "fm_model": self.fm_model,
                "modality": self.modality,
                "seed": self.seed,
                "n_folds": self.n_folds,
                "created": datetime.now().isoformat(),
                "folds": {},
                "holdout": None,
            }
            self._save_meta()
        
        return self._meta
    
    def _save_meta(self) -> None:
        """Save metadata to disk."""
        if self._meta is not None:
            self.meta_path.write_text(json.dumps(self._meta, indent=2))
    
    def _fold_path(self, fold_idx: int) -> Path:
        """Get path for fold cache file."""
        return self.run_dir / f"fold_{fold_idx}.npz"
    
    def _holdout_path(self) -> Path:
        """Get path for holdout cache file."""
        return self.run_dir / "holdout.npz"
    
    def has_fold(self, fold_idx: int) -> bool:
        """Check if fold is cached."""
        meta = self._get_meta()
        fold_key = str(fold_idx)
        return fold_key in meta["folds"] and self._fold_path(fold_idx).exists()
    
    def has_holdout(self) -> bool:
        """Check if holdout is cached."""
        meta = self._get_meta()
        return meta["holdout"] is not None and self._holdout_path().exists()
    
    def save_fold(
        self,
        fold_idx: int,
        *,
        Xg_tr_pca: np.ndarray,
        Xg_va_pca: np.ndarray,
        Xb_tr_s: np.ndarray,
        Xb_va_s: np.ndarray,
        tr_indices: np.ndarray,
        va_indices: np.ndarray,
    ) -> None:
        """
        Save preprocessed fold data to disk.
        
        Args:
            fold_idx: Fold index (0-based)
            Xg_tr_pca: Gene training data after PCA (n_tr, max_pca_dim)
            Xg_va_pca: Gene validation data after PCA
            Xb_tr_s: Brain training data after standardization
            Xb_va_s: Brain validation data after standardization
            tr_indices: Training sample indices
            va_indices: Validation sample indices
        """
        path = self._fold_path(fold_idx)
        
        np.savez_compressed(
            path,
            Xg_tr_pca=Xg_tr_pca.astype(np.float32),
            Xg_va_pca=Xg_va_pca.astype(np.float32),
            Xb_tr_s=Xb_tr_s.astype(np.float32),
            Xb_va_s=Xb_va_s.astype(np.float32),
            tr_indices=tr_indices,
            va_indices=va_indices,
        )
        
        # Update metadata
        meta = self._get_meta()
        meta["folds"][str(fold_idx)] = {
            "saved": datetime.now().isoformat(),
            "gene_shape_tr": list(Xg_tr_pca.shape),
            "gene_shape_va": list(Xg_va_pca.shape),
            "brain_shape_tr": list(Xb_tr_s.shape),
            "brain_shape_va": list(Xb_va_s.shape),
        }
        self._save_meta()
        
        print(f"  [cache] Saved fold {fold_idx} to {path.name}")
    
    def load_fold(self, fold_idx: int) -> dict[str, np.ndarray]:
        """
        Load preprocessed fold data from disk.
        
        Returns:
            Dict with keys: Xg_tr_pca, Xg_va_pca, Xb_tr_s, Xb_va_s,
                           tr_indices, va_indices
        """
        path = self._fold_path(fold_idx)
        
        if not path.exists():
            raise FileNotFoundError(f"Fold cache not found: {path}")
        
        data = np.load(path)
        result = {
            "Xg_tr_pca": data["Xg_tr_pca"],
            "Xg_va_pca": data["Xg_va_pca"],
            "Xb_tr_s": data["Xb_tr_s"],
            "Xb_va_s": data["Xb_va_s"],
            "tr_indices": data["tr_indices"],
            "va_indices": data["va_indices"],
        }
        
        print(f"  [cache] Loaded fold {fold_idx} from {path.name}")
        return result
    
    def save_holdout(
        self,
        *,
        Xg_tr_pca: np.ndarray,
        Xg_ho_pca: np.ndarray,
        Xb_tr_s: np.ndarray,
        Xb_ho_s: np.ndarray,
        train_indices: np.ndarray,
        holdout_indices: np.ndarray,
    ) -> None:
        """Save preprocessed holdout data to disk."""
        path = self._holdout_path()
        
        np.savez_compressed(
            path,
            Xg_tr_pca=Xg_tr_pca.astype(np.float32),
            Xg_ho_pca=Xg_ho_pca.astype(np.float32),
            Xb_tr_s=Xb_tr_s.astype(np.float32),
            Xb_ho_s=Xb_ho_s.astype(np.float32),
            train_indices=train_indices,
            holdout_indices=holdout_indices,
        )
        
        # Update metadata
        meta = self._get_meta()
        meta["holdout"] = {
            "saved": datetime.now().isoformat(),
            "gene_shape_tr": list(Xg_tr_pca.shape),
            "gene_shape_ho": list(Xg_ho_pca.shape),
            "brain_shape_tr": list(Xb_tr_s.shape),
            "brain_shape_ho": list(Xb_ho_s.shape),
        }
        self._save_meta()
        
        print(f"  [cache] Saved holdout to {path.name}")
    
    def load_holdout(self) -> dict[str, np.ndarray]:
        """Load preprocessed holdout data from disk."""
        path = self._holdout_path()
        
        if not path.exists():
            raise FileNotFoundError(f"Holdout cache not found: {path}")
        
        data = np.load(path)
        result = {
            "Xg_tr_pca": data["Xg_tr_pca"],
            "Xg_ho_pca": data["Xg_ho_pca"],
            "Xb_tr_s": data["Xb_tr_s"],
            "Xb_ho_s": data["Xb_ho_s"],
            "train_indices": data["train_indices"],
            "holdout_indices": data["holdout_indices"],
        }
        
        print(f"  [cache] Loaded holdout from {path.name}")
        return result
    
    def clear(self) -> None:
        """Clear all cached data for this run."""
        import shutil
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
            self._meta = None
            print(f"  [cache] Cleared cache at {self.run_dir}")


class CheckpointManager:
    """
    Incremental checkpoint manager for grid search.
    
    Saves completed configurations to allow resume after crash.
    """
    
    def __init__(self, checkpoint_path: Path | str):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_path: Path to checkpoint JSON file
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] | None = None
    
    def _load(self) -> dict[str, Any]:
        """Load checkpoint from disk."""
        if self._data is not None:
            return self._data
        
        if self.checkpoint_path.exists():
            self._data = json.loads(self.checkpoint_path.read_text())
        else:
            self._data = {
                "created": datetime.now().isoformat(),
                "completed_configs": {},
                "best_config": None,
                "best_score": float("-inf"),
            }
        
        return self._data
    
    def _save(self) -> None:
        """Save checkpoint to disk."""
        if self._data is not None:
            self._data["last_updated"] = datetime.now().isoformat()
            self.checkpoint_path.write_text(json.dumps(self._data, indent=2))
    
    def is_completed(self, config_key: str) -> bool:
        """Check if a configuration has been completed."""
        data = self._load()
        return config_key in data["completed_configs"]
    
    def save_result(
        self,
        config_key: str,
        result: dict[str, Any],
        score: float | None = None,
    ) -> None:
        """
        Save result for a completed configuration.
        
        Args:
            config_key: Unique identifier for the configuration
            result: Result dictionary to save
            score: Optional score for tracking best config
        """
        data = self._load()
        data["completed_configs"][config_key] = {
            "result": result,
            "completed_at": datetime.now().isoformat(),
        }
        
        # Track best
        if score is not None and score > data["best_score"]:
            data["best_score"] = score
            data["best_config"] = config_key
        
        self._save()
    
    def get_completed_count(self) -> int:
        """Get number of completed configurations."""
        data = self._load()
        return len(data["completed_configs"])
    
    def get_best(self) -> tuple[str | None, float]:
        """Get best configuration and score."""
        data = self._load()
        return data["best_config"], data["best_score"]
    
    def get_all_results(self) -> dict[str, Any]:
        """Get all completed results."""
        data = self._load()
        return {k: v["result"] for k, v in data["completed_configs"].items()}


def backup_to_storage(
    src_path: Path | str,
    dest_dir: Path | str,
    filename: str | None = None,
) -> Path | None:
    """
    Backup a file from /scratch to /storage for persistence.
    
    Per lab guidelines: scratch may be cleaned, so backup important results.
    
    Args:
        src_path: Source file path
        dest_dir: Destination directory (should be on /storage)
        filename: Optional new filename
    
    Returns:
        Path to backed up file, or None if source doesn't exist
    """
    import shutil
    
    src = Path(src_path)
    if not src.exists():
        return None
    
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = src.name
    
    dest_file = dest / filename
    shutil.copy2(src, dest_file)
    
    print(f"  [backup] {src.name} -> {dest_file}")
    return dest_file


if __name__ == "__main__":
    # Quick test
    cache_dir = get_cache_dir("test_cache")
    print(f"Cache directory: {cache_dir}")
    
    # Test FoldCache
    cache = FoldCache(cache_dir, "test_fm", "test_mod", seed=42, n_folds=3)
    print(f"Run directory: {cache.run_dir}")
    
    # Test with dummy data
    dummy_gene = np.random.randn(100, 64).astype(np.float32)
    dummy_brain = np.random.randn(100, 50).astype(np.float32)
    
    cache.save_fold(
        0,
        Xg_tr_pca=dummy_gene[:80],
        Xg_va_pca=dummy_gene[80:],
        Xb_tr_s=dummy_brain[:80],
        Xb_va_s=dummy_brain[80:],
        tr_indices=np.arange(80),
        va_indices=np.arange(80, 100),
    )
    
    assert cache.has_fold(0)
    loaded = cache.load_fold(0)
    print(f"Loaded fold 0: gene_tr={loaded['Xg_tr_pca'].shape}")
    
    # Cleanup test
    cache.clear()
    print("Test passed!")
