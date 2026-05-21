# cartm/metrics/stratified_perplexity.py
import jax.numpy as jnp
from jax import Array
from cartm.metrics.metric_base import Metric
from cartm.core import EPSILON


class StratifiedPerplexityMetric(Metric):
    """
    Стратифицированная перплексия по темам (токен-уровневая аппроксимация).
    Совместима с базовым интерфейсом Metric из вашего проекта.
    """
    def __init__(self, tag: str | None = None, n_topics: int = 10):
        # Вызываем конструктор родителя строго по сигнатуре Metric.__init__(tag: str)
        super().__init__(tag=tag or "stratified_perplexity")
        self.n_topics = n_topics
        self._topic_ll = jnp.zeros(n_topics)
        self._topic_counts = jnp.zeros(n_topics)
        self._last_per_topic = None
        self._per_topic_hist = []

    def partial_update(
        self,
        *,
        batch: Array,
        phi: Array,
        theta: Array,
    ):
        # p(w_i | C_i) = \sum_t p(w_i|t) * p(t|C_i)
        p_wi = jnp.sum(theta * phi[batch], axis=1)
        log_probs = jnp.log(p_wi + EPSILON)
        
        # Доминирующая тема для каждого токена
        dominant_topics = jnp.argmax(theta, axis=1)

        self._topic_ll = self._topic_ll.at[dominant_topics].add(log_probs)
        self._topic_counts = self._topic_counts.at[dominant_topics].add(
            jnp.ones_like(log_probs)
        )

    def _flush(self) -> float:
        # Защита от деления на ноль
        safe_counts = jnp.where(self._topic_counts > 0, self._topic_counts, 1.0)
        
        # Per-topic perplexity
        topic_ppls = jnp.exp(-self._topic_ll / safe_counts)
        
        # Исключаем темы без данных из усреднения
        valid_mask = self._topic_counts > 0
        valid_ppls = jnp.where(valid_mask, topic_ppls, jnp.nan)
        macro_ppl = jnp.nanmean(valid_ppls)
        
        # Сохраняем per-topic значения для визуализации
        self._last_per_topic = {
            f"topic_{t}": float(topic_ppls[t]) if valid_mask[t] else None
            for t in range(self.n_topics)
        }
        self._per_topic_hist.append(topic_ppls.tolist())

        # Сброс накопителей
        self._topic_ll = jnp.zeros(self.n_topics)
        self._topic_counts = jnp.zeros(self.n_topics)
        
        return float(macro_ppl) if not jnp.isnan(macro_ppl) else 0.0

    def get_per_topic_values(self) -> dict[str, float | None]:
        """Возвращает per-topic PPL последней эпохи для экспорта/визуализации."""
        return self._last_per_topic or {}

    @property
    def per_topic_history(self) -> list:
        """Per-topic PPL history."""
        return self._per_topic_hist
