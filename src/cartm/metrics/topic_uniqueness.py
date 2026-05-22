# cartm/metrics/topic_uniqueness.py
"""
Метрика уникальности тем по топ-словам (per-topic uniqueness).

Для каждой темы t вычисляется доля её топ-k слов, не встречающихся
в топ-k словах других тем:

    U_t = 1 - |W_t ∩ (union_{j != t} W_j)| / k

где W_t — множество k слов с наибольшими значениями phi[:, t].

Метрика информативна для несбалансированных коллекций: у редкой темы,
лексически поглощаемой доминирующими темами, U_t стремится к нулю,
тогда как агрегированное разнообразие тем может оставаться высоким
за счёт частых тем.

Класс реализован по образцу NPMICoherenceMetric и TopicVarianceMetric:
в partial_update запоминается последняя матрица phi, в _flush по ней
вычисляются значения U_t. Расчёт не требует ни внешней разметки,
ни опорного корпуса.

Интерфейс согласован с базовым классом Metric: метод flush() заносит
в history() единственное число — среднее U_t по темам (macro-uniqueness).
Значения U_t по отдельным темам доступны через свойство
per_topic_history (по аналогии со StratifiedPerplexityMetric).
"""

import numpy as np
import jax.numpy as jnp
from jax import Array

from cartm.metrics.metric_base import Metric


class TopicUniquenessMetric(Metric):
    """Уникальность тем по топ-словам (per-topic uniqueness)."""

    def __init__(self, top_k: int = 10, tag: str | None = None):
        """
        Args:
            top_k: число топ-слов темы, по которым считается уникальность.
            tag: имя метрики для отображения в логах.
        """
        super().__init__(tag=tag or "topic_uniqueness")
        self.top_k = top_k

        self._last_phi = None
        self._last_per_topic = None
        self._per_topic_hist = []

    def partial_update(
        self,
        *,
        batch: Array,
        phi: Array,
        theta: Array,
    ):
        # Запоминается последняя матрица phi = p(w|t), (W, T).
        # По аналогии с NPMICoherenceMetric расчёт откладывается на _flush.
        self._last_phi = phi

    def _flush(self) -> float:
        if self._last_phi is None:
            return 0.0

        phi = np.asarray(self._last_phi)  # (W, T)
        n_topics = phi.shape[1]
        k = self.top_k

        # Топ-k индексов слов для каждой темы -> матрица (T, k).
        top_idx = np.argpartition(phi, -k, axis=0)[-k:]  # (k, T)
        top_idx = top_idx.T  # (T, k)

        # Для каждого слова, попавшего хотя бы в один топ, считаем,
        # в скольких темах оно является топовым.
        flat = top_idx.reshape(-1)
        word_topic_count = {}
        for w in flat:
            word_topic_count[w] = word_topic_count.get(w, 0) + 1

        # U_t = доля топ-слов темы, не разделяемых с другими темами.
        # Слово уникально для темы, если оно топовое ровно в одной теме.
        per_topic = np.zeros(n_topics, dtype=np.float32)
        for t in range(n_topics):
            exclusive = sum(
                1 for w in top_idx[t] if word_topic_count[w] == 1
            )
            per_topic[t] = exclusive / k

        self._last_per_topic = {
            f"topic_{t}": float(per_topic[t]) for t in range(n_topics)
        }
        self._per_topic_hist.append(per_topic.tolist())

        self._last_phi = None

        return float(np.mean(per_topic))

    def get_per_topic_values(self) -> dict[str, float]:
        """Возвращает U_t по темам для последней эпохи."""
        return self._last_per_topic or {}

    @property
    def per_topic_history(self) -> list:
        """История значений U_t по темам (список по эпохам)."""
        return self._per_topic_hist
