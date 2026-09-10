"""Audio feature reranker for search result refinement."""

import numpy as np
from loguru import logger

from core.config import get_settings


class AudioFeatureReranker:
    """Reranks search results using audio feature similarity."""

    SCALAR_FEATURES = [
        "tempo",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "rms_energy_mean",
        "zero_crossing_rate_mean",
        "spectral_bandwidth_mean",
    ]

    def __init__(self, weights: dict[str, float] | None = None):
        """Initialize reranker with feature weights.

        Args:
            weights: Custom feature weights. Defaults to RerankerSettings.
        """
        if weights is not None:
            self.weights = weights
        else:
            settings = get_settings().reranker
            self.weights = {
                "cosine": settings.cosine_weight,
                "tempo": settings.tempo_weight,
                "chroma": settings.chroma_weight,
                "spectral": settings.spectral_weight,
                "energy": settings.energy_weight,
            }

    def rerank(
        self,
        query_features: dict[str, float | list[float]],
        results: list[dict],
        top_n: int | None = None,
    ) -> list[dict]:
        """Rerank search results by combining cosine similarity with audio features.

        Args:
            query_features: Audio features of the query track.
            results: Search results from ChromaDB, each with "distance" and "metadata".
            top_n: Number of results to return. Defaults to len(results).

        Returns:
            Results sorted by reranked score (descending), trimmed to top_n.
        """
        if not results:
            return results

        if top_n is None:
            top_n = len(results)

        scored = []
        for result in results:
            score = self._compute_fused_score(query_features, result)
            result["reranked_score"] = score
            scored.append(result)

        scored.sort(key=lambda r: r["reranked_score"], reverse=True)
        return scored[:top_n]

    def _compute_fused_score(
        self,
        query_features: dict[str, float | list[float]],
        result: dict,
    ) -> float:
        """Compute weighted fused similarity score for a single result."""
        metadata = result.get("metadata", {})

        cosine_sim = 1.0 - result.get("distance", 1.0)
        has_features = (
            any(k in metadata for k in self.SCALAR_FEATURES)
            or "chroma_mean" in metadata
        )

        if not has_features:
            logger.debug(
                f"No feature metadata for {result.get('id', '?')}, using cosine only"
            )
            return cosine_sim

        parts = {"cosine": cosine_sim}

        for feat in self.SCALAR_FEATURES:
            if feat in metadata and feat in query_features:
                q_val = query_features[feat]
                c_val = metadata[feat]
                parts[feat] = self._scalar_similarity(
                    q_val, c_val, feat, query_features
                )

        if "chroma_mean" in metadata and "chroma_mean" in query_features:
            parts["chroma"] = self._chroma_similarity(
                query_features["chroma_mean"], metadata["chroma_mean"]
            )

        spectral_keys = {
            "spectral_centroid_mean",
            "spectral_centroid_std",
            "spectral_bandwidth_mean",
        }
        spectral_vals = [parts[k] for k in spectral_keys if k in parts]
        if spectral_vals:
            parts["spectral"] = float(np.mean(spectral_vals))

        energy_keys = {"rms_energy_mean", "zero_crossing_rate_mean"}
        energy_vals = [parts[k] for k in energy_keys if k in parts]
        if energy_vals:
            parts["energy"] = float(np.mean(energy_vals))

        total_weight = 0.0
        weighted_sum = 0.0
        for key, weight in self.weights.items():
            if key in parts:
                weighted_sum += weight * parts[key]
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else cosine_sim

    def _scalar_similarity(
        self,
        query_val: float,
        candidate_val: float,
        feat_name: str,
        query_features: dict,
    ) -> float:
        """Compute similarity for a scalar feature using absolute difference."""
        return 1.0 - abs(query_val - candidate_val) / max(
            abs(query_val), abs(candidate_val), 1e-8
        )

    @staticmethod
    def _chroma_similarity(
        query_chroma: list[float], candidate_chroma: list[float]
    ) -> float:
        """Compute cosine similarity between two chroma vectors."""
        q = np.array(query_chroma, dtype=np.float64)
        c = np.array(candidate_chroma, dtype=np.float64)
        norm_q = np.linalg.norm(q)
        norm_c = np.linalg.norm(c)
        if norm_q < 1e-8 or norm_c < 1e-8:
            return 0.0
        return float(np.dot(q, c) / (norm_q * norm_c))
