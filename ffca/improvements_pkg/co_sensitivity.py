"""CoSensitivityGroups — gradient-correlation k-medoids with guardrails (audit-v2)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

from ..core.archetypes import is_noise_candidate


class CoSensitivityGroups:
    def __init__(
        self,
        k: int | None = None,
        nc_threshold: float = 0.5,
        review_threshold: float = 0.3,
        n_permutations: int = 100,
        n_bootstrap: int = 30,
        seed: int = 42,
    ):
        self.k = k
        self.nc_threshold = nc_threshold
        self.review_threshold = review_threshold
        self.n_permutations = n_permutations
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.results: dict = {}
        self.diagnostics: dict = {}

    @staticmethod
    def _grad_correlation(g: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.corrcoef(g.T), nan=0.0)

    @staticmethod
    def _kmedoids(dist: np.ndarray, k: int, max_iter: int = 30, seed: int = 42):
        rng = np.random.default_rng(seed)
        n = dist.shape[0]
        first = int(rng.integers(0, n))
        medoids = [first]
        for _ in range(k - 1):
            d_to_med = dist[:, medoids].min(axis=1)
            d_to_med[medoids] = -1
            medoids.append(int(np.argmax(d_to_med)))
        medoids = np.array(medoids)
        for _ in range(max_iter):
            labels = np.argmin(dist[:, medoids], axis=1)
            new_medoids = medoids.copy()
            for cl in range(k):
                members = np.where(labels == cl)[0]
                if len(members) == 0:
                    continue
                sub = dist[np.ix_(members, members)]
                new_medoids[cl] = members[np.argmin(sub.sum(axis=1))]
            if np.all(new_medoids == medoids):
                break
            medoids = new_medoids
        return medoids, np.argmin(dist[:, medoids], axis=1)

    def _select_k(self, dist: np.ndarray, k_range):
        best_k, best_score = next(iter(k_range)), -np.inf
        for k in k_range:
            if k < 2 or k >= dist.shape[0]:
                continue
            _, labels = self._kmedoids(dist, k, seed=self.seed)
            if len(set(labels)) < 2:
                continue
            try:
                s = silhouette_score(dist, labels, metric="precomputed")
            except ValueError:
                continue
            if s > best_score:
                best_k, best_score = k, s
        return best_k

    def _permutation_silhouette(self, g: np.ndarray, k: int, observed: float):
        rng = np.random.default_rng(self.seed)
        scores = []
        for _ in range(self.n_permutations):
            perm = g.copy()
            for j in range(perm.shape[1]):
                rng.shuffle(perm[:, j])
            c = self._grad_correlation(perm)
            d_ = np.clip(1.0 - np.abs(c), 0, 1)
            np.fill_diagonal(d_, 0)
            try:
                _, labels = self._kmedoids(d_, k, seed=self.seed)
                if len(set(labels)) >= 2:
                    scores.append(silhouette_score(d_, labels, metric="precomputed"))
            except Exception:
                continue
        if not scores:
            return 1.0, float("nan")
        return float((np.asarray(scores) >= observed).mean()), float(np.quantile(scores, 0.95))

    def _bootstrap_ari(self, g: np.ndarray, k: int, ref):
        rng = np.random.default_rng(self.seed + 1)
        n = g.shape[0]
        boot = max(int(n * 0.8), 5)
        scores = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=boot, replace=True)
            c = self._grad_correlation(g[idx])
            d_ = np.clip(1.0 - np.abs(c), 0, 1)
            np.fill_diagonal(d_, 0)
            try:
                _, labels = self._kmedoids(d_, k, seed=self.seed)
                scores.append(adjusted_rand_score(ref, labels))
            except Exception:
                continue
        return float(np.median(scores)) if scores else float("nan")

    def compute(
        self,
        gradients: np.ndarray | None = None,
        gradient_correlation: np.ndarray | None = None,
        impact: np.ndarray | None = None,
        volatility: np.ndarray | None = None,
        nonlinearity: np.ndarray | None = None,
        interaction: np.ndarray | None = None,
        run_guardrails: bool = True,
    ) -> dict:
        if gradient_correlation is None and gradients is None:
            raise ValueError("provide gradients or gradient_correlation")
        if gradient_correlation is None:
            gradient_correlation = self._grad_correlation(gradients)
        d = gradient_correlation.shape[0]
        dist = np.clip(1.0 - np.abs(gradient_correlation), 0, 1)
        np.fill_diagonal(dist, 0)

        k = self.k or self._select_k(dist, range(2, min(8, d // 5 + 3)))
        medoids, labels = self._kmedoids(dist, k, seed=self.seed)
        obs_sil = silhouette_score(dist, labels, metric="precomputed") \
            if len(set(labels)) >= 2 else 0.0

        p_perm = null_95 = ari_med = float("nan")
        if run_guardrails and gradients is not None:
            p_perm, null_95 = self._permutation_silhouette(gradients, k, obs_sil)
            ari_med = self._bootstrap_ari(gradients, k, labels)

        if impact is None or volatility is None or nonlinearity is None or interaction is None:
            is_noise = np.zeros(d, dtype=bool)
        else:
            is_noise = is_noise_candidate(impact, volatility, nonlinearity, interaction)

        groups = {}
        for c in range(k):
            mask = labels == c
            size = int(mask.sum())
            nc_frac = float(is_noise[mask].mean()) if size else 0.0
            if nc_frac > self.nc_threshold:
                rec = "PRUNE — noise-dominated group"
            elif nc_frac > self.review_threshold:
                rec = "REVIEW — significant noise"
            else:
                rec = "KEEP — mostly useful"
            full_members = np.where(mask)[0].tolist()
            groups[c] = {
                "size": size,
                "medoid": int(medoids[c]),
                "channels": full_members[:10],       # legacy: first 10 only
                "channels_full": full_members,        # v0.7: complete list
                "nc_fraction": round(nc_frac, 3),
                "mean_impact": round(float(np.asarray(impact)[mask].mean()), 4)
                    if impact is not None and size else None,
                "mean_interaction": round(float(np.asarray(interaction)[mask].mean()), 4)
                    if interaction is not None and size else None,
                "recommendation": rec,
            }

        best_nc = max((g["nc_fraction"] for g in groups.values()), default=0.0)
        abort = best_nc < self.nc_threshold
        passes_perm = (not run_guardrails) or np.isnan(p_perm) or p_perm < 0.05
        passes_ari = (not run_guardrails) or np.isnan(ari_med) or ari_med >= 0.5

        # `abort_recommended` is True when Co-Sens declines to recommend
        # any prune — either no group is noise-dominated, or the
        # clustering itself fails the permutation/bootstrap gates. Name
        # kept for backward compatibility with existing report.json
        # consumers; the positively-named `prune_safe_group_found` is
        # the inverted form and is what new code should prefer.
        prune_unsafe = bool(abort or not passes_perm or not passes_ari)
        self.diagnostics = {
            "k": k,
            "silhouette_observed": float(obs_sil),
            "silhouette_null_95": null_95,
            "permutation_p": p_perm,
            "bootstrap_ari_median": ari_med,
            "best_nc_fraction": best_nc,
            "abort_recommended": prune_unsafe,
            "prune_safe_group_found": not prune_unsafe,
        }
        self.results = groups
        return groups

    def summary(self) -> dict:
        total = sum(g["size"] for g in self.results.values())
        prunable = sum(g["size"] for g in self.results.values()
                       if g["recommendation"].startswith("PRUNE"))
        return {
            "total_features": total,
            "prunable": prunable,
            "n_groups": len(self.results),
            **self.diagnostics,
        }
