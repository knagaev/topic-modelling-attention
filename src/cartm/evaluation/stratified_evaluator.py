# cartm/evaluation/stratified_evaluator.py
import jax.numpy as jnp
import numpy as np
from typing import Iterable, Dict, Optional
from cartm.core import EPSILON


class ExternalStratifiedEvaluator:
    """
    Вычисляет документ-уровневую стратифицированную перплексию постфактум.
    
    Не модифирует состояние модели и не требует изменений интерфейса Metric.
    Группирует документы по доминирующей теме (argmax(mean(theta_doc)))
    и вычисляет PPL для каждой темы отдельно, затем macro-average.
    """
    def __init__(self, model, num_attn_passes: int = 1):
        self.model = model
        self.n_topics = model.n_topics
        self.num_attn_passes = num_attn_passes
        # Фиксируем параметры модели на момент инициализации
        self._phi = model.phi
        self._n_t = model.n_t
        self._p_w = model.p_w
        self._ctx_weights = model.context_weights
        # Предрассчитываем phi_wt = p(w|t)
        self._phi_wt = model.renormalize_phi(p_w=self._p_w, phi=self._phi)

    def evaluate(
        self, 
        batches: Iterable[tuple[jnp.ndarray, jnp.ndarray]]
    ) -> Dict:
        """
        Прогоняет батчи через модель в inference-режиме и вычисляет стратифицированную PPL.
        
        Args:
            batches: Iterable[tuple[tokens_batch, ctx_bounds_batch]]
            
        Returns:
            dict с macro_ppl, per_topic_ppl, topic_doc_counts
        """
        topic_ll = {t: 0.0 for t in range(self.n_topics)}
        topic_len = {t: 0 for t in range(self.n_topics)}
        topic_doc_count = {t: 0 for t in range(self.n_topics)}

        for batch, ctx_bounds in batches:
            # 🔹 Inference: вычисляем theta без обновления весов
            theta, _, _, _ = self.model._step(
                batch=batch,
                ctx_bounds=ctx_bounds,
                phi=self._phi,
                n_t=self._n_t,
                ctx_weights=self._ctx_weights,
                num_attn_passes=self.num_attn_passes
            )

            # p(w_i | C_i) = sum_t p(w_i|t) * p(t|C_i)
            p_wi = jnp.sum(theta * self._phi_wt[batch], axis=1)
            log_probs = jnp.log(p_wi + EPSILON)

            # Переходим в numpy для безопасного slicing по границам документов
            log_probs_np = np.asarray(log_probs)
            theta_np = np.asarray(theta)
            bounds_np = np.asarray(ctx_bounds)

            start = 0
            for bound in bounds_np:
                bound = int(bound)
                if bound <= start:
                    continue

                # Статистика по документу
                doc_log_ll = np.sum(log_probs_np[start:bound])
                doc_len = bound - start
                doc_theta_mean = np.mean(theta_np[start:bound], axis=0)
                dominant_topic = int(np.argmax(doc_theta_mean))

                topic_ll[dominant_topic] += doc_log_ll
                topic_len[dominant_topic] += doc_len
                topic_doc_count[dominant_topic] += 1

                start = bound

            # Обработка остатка батча (если границы не покрывают все токены)
            if start < len(log_probs_np):
                doc_log_ll = np.sum(log_probs_np[start:])
                doc_len = len(log_probs_np) - start
                doc_theta_mean = np.mean(theta_np[start:], axis=0)
                dominant_topic = int(np.argmax(doc_theta_mean))

                topic_ll[dominant_topic] += doc_log_ll
                topic_len[dominant_topic] += doc_len
                topic_doc_count[dominant_topic] += 1

        # 📊 Агрегация: per-topic PPL → macro-average
        per_topic_ppl = {}
        valid_ppls = []
        for t in range(self.n_topics):
            if topic_len[t] > 0:
                ppl = float(np.exp(-topic_ll[t] / topic_len[t]))
                per_topic_ppl[f"topic_{t}"] = ppl
                valid_ppls.append(ppl)
            else:
                per_topic_ppl[f"topic_{t}"] = None

        macro_ppl = float(np.mean(valid_ppls)) if valid_ppls else 0.0

        return {
            "macro_stratified_ppl": macro_ppl,
            "per_topic_ppl": per_topic_ppl,
            "topic_doc_counts": topic_doc_count,
            "total_docs": sum(topic_doc_count.values())
        }