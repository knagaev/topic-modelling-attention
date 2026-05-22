"""
exp_stand_abl.py — экспериментальный стенд для абляционного исследования
члена N_wt в модели AARTM.

Назначение
----------
Скрипт повторяет логику exp_stand.py, но позволяет выбирать вариант модели:
    - 'full'   : полная модель AttentiveTopicModel (с членом N_wt в M-шаге);
    - 'no_nwt' : абляционный вариант без члена N_wt.

Сравнение результатов двух вариантов при идентичных прочих условиях
обеспечивает проверку гипотезы Г2 (вклад контекстного члена N_wt в
когерентность тем).

Важно о реализации абляции
--------------------------
Класс AttentiveTopicModelNoNWT из experiments/model_no_N_wt.py имеет
сигнатуры _step и _update_phi, несовместимые с базовым методом обучения
fit_with_test (он используется в exp_stand.py). Поэтому абляционный
вариант определён здесь локально — подклассом AttentiveTopicModel,
который переопределяет только _step, сохраняя сигнатуру базового класса
(4 возвращаемых значения, без grad_reg) и обнуляя счётчик N_wt.
В результате методы _update_phi, _batched_step_wrapper и fit_with_test
остаются базовыми, а член N_wt не вносит вклад в M-шаг — что и требуется
для абляции.

Запуск
------
    python exp_stand_abl.py STRUCTURED_DATA_PREFIX EXP_MODE MODEL_VARIANT

    STRUCTURED_DATA_PREFIX : ['full', 'imbalanced_test', 'minus_rel', 'minus_med']
    EXP_MODE               : ['base', 'gamma', 'n_topics_gamma']
    MODEL_VARIANT          : ['full', 'no_nwt']   (по умолчанию 'full')

Имена выходных файлов содержат метку варианта модели, чтобы результаты
полной и абляционной моделей не перезаписывали друг друга:
    result_n_topics_{nt}_gamma_{g}_ctx_len_{cl}_{dataset}_{variant}.json

Метрики
-------
Помимо метрик базового exp_stand.py (perplexity, npmi_coherence,
stratified_perplexity, stratified_pt_perplexity) скрипт вычисляет
метрику уникальности тем (per-topic uniqueness):
    topic_uniqueness     — среднее U_t по темам по эпохам;
    topic_pt_uniqueness  — значения U_t по отдельным темам по эпохам.

ВАЖНО. Скрипт импортирует класс TopicUniquenessMetric из модуля
cartm.metrics.topic_uniqueness. Перед запуском поместите файл
topic_uniqueness.py в каталог пакета src/cartm/metrics/ рядом с
остальными метриками. Регистрировать класс в metrics/__init__.py
не обязательно — здесь используется прямой импорт из модуля.
"""

import pickle
import sys
import json

import jax
import jax.numpy as jnp

from cartm import ContextTopicModel, AttentiveTopicModel
from cartm.core import EPSILON, norm, calc_attn
from cartm.preprocessing import (
    CorpusLoader,
    BatchedCorpusLoader,
    build_bow,
    get_structured_data
)
from cartm.metrics import (
    PerplexityMetric,
    NPMICoherenceMetric,
    SparsityMetric,
    TopicVarianceMetric,
    StratifiedPerplexityMetric
)
# Метрика per-topic uniqueness. Файл topic_uniqueness.py должен быть
# помещён в каталог пакета cartm/metrics/ рядом с остальными метриками
# (см. инструкцию в шапке файла и комментарий ниже).
from cartm.metrics.topic_uniqueness import TopicUniquenessMetric
from cartm.regularization import DecorrelationRegularization


# ----------------------------------------------------------------------
# Абляционный вариант модели: AARTM без члена N_wt.
#
# Переопределяется только _step. Сигнатура совпадает с базовым
# AttentiveTopicModel._step (см. cartm/aartm.py): принимает
# (batch, ctx_bounds, phi, n_t, ctx_weights, num_attn_passes),
# возвращает (theta, n_t_new, n_wt, N_wt).
#
# Отличие от базового _step: счётчик N_wt не вычисляется через
# calc_attn_transposed, а полагается нулевым. Поскольку базовый
# M-шаг _update_phi прибавляет слагаемое coeff * N_wt, обнуление
# N_wt в точности убирает вклад контекстного члена из обновления Phi.
# ----------------------------------------------------------------------
class AttentiveTopicModelAblNoNWT(AttentiveTopicModel):
    """AARTM без контекстного члена N_wt (абляция, совместимая с fit_with_test)."""

    @staticmethod
    @jax.jit(static_argnames=("num_attn_passes"))
    def _step(
        batch: jax.Array,
        ctx_bounds: jax.Array,
        phi: jax.Array,
        n_t: jax.Array,
        ctx_weights: jax.Array,
        num_attn_passes: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        p_it = norm(phi[batch], axis=1)  # (I, T)

        for _ in range(num_attn_passes):
            theta = calc_attn(
                matrix=p_it,
                ctx_bounds=ctx_bounds,
                ctx_weights=ctx_weights,
            )  # (I, T)
            p_it = norm(p_it * theta / (n_t + EPSILON), axis=1)  # (I, T)

        n_t_new = jnp.sum(p_it, axis=0)  # (T,)
        n_wt = jax.ops.segment_sum(p_it, batch, phi.shape[0])  # (W, T)

        # Абляция: член N_wt не вычисляется, полагается нулевым.
        N_wt = jnp.zeros_like(phi)  # (W, T)

        return theta, n_t_new, n_wt, N_wt


# Сопоставление метки варианта классу модели.
MODEL_VARIANTS = {
    'full': AttentiveTopicModel,
    'no_nwt': AttentiveTopicModelAblNoNWT,
}


CATEGORIES_QTY = 20
CATEGORY_MED = 13  # sci.med
CATEGORY_REL = 15  # soc.religion.christian
CORPUS_FILENAME = 'corpus_data.pkl'
STRUCTURED_DATA_FILENAME = 'structured_data.pkl'
VOCAB_FILENAME = 'corpus_vocab.json'

CORPUS_PATH = 'data/corpuses/'
STRUCTURED_DATA_PATH = 'data/structured/'
RESULT_PATH = 'data/results/'


if len(sys.argv) < 3:
    print('Usage: python exp_stand_abl.py STRUCTURED_DATA_PREFIX EXP_MODE [MODEL_VARIANT]')
    print("  STRUCTURED_DATA_PREFIX : ['full', 'imbalanced_test', 'minus_rel', 'minus_med']")
    print("  EXP_MODE               : ['base', 'gamma', 'n_topics_gamma']")
    print("  MODEL_VARIANT          : ['full', 'no_nwt']  (default: 'full')")
    exit()

corpus_data_prefix = 'full'
structured_data_prefix = sys.argv[1]  # датасет
exp_mode = sys.argv[2]                # режим эксперимента
model_variant = sys.argv[3] if len(sys.argv) > 3 else 'full'  # вариант модели

if model_variant not in MODEL_VARIANTS:
    print(f"Unknown MODEL_VARIANT '{model_variant}'. Use one of: {list(MODEL_VARIANTS)}")
    exit()

print(f"{structured_data_prefix=}")
print(f"{exp_mode=}")
print(f"{model_variant=}")

model_cls = MODEL_VARIANTS[model_variant]


structured_data_filepath = (
    STRUCTURED_DATA_PATH + structured_data_prefix + '_' + STRUCTURED_DATA_FILENAME
)

with open(structured_data_filepath, 'rb') as file:
    structured_data = pickle.load(file)
    print('structured_data ok')

(train_tokens, train_bounds, train_result_structure), (
 test_tokens, test_bounds, test_result_structure) = structured_data

print(f"{train_result_structure=}")
print(f"{test_result_structure=}")

vocab_filepath = CORPUS_PATH + corpus_data_prefix + '_' + VOCAB_FILENAME
with open(vocab_filepath, 'r') as file:
    vocab = json.load(file)
    vocab_size = len(vocab)

train_loader = BatchedCorpusLoader(
    data=train_tokens,
    doc_bounds=train_bounds,
    batch_size=10000,
)

test_loader = BatchedCorpusLoader(
    data=test_tokens,
    doc_bounds=test_bounds,
    batch_size=10000,
)

experiments = {
    'base': {
        'n_topics': [20],
        'gamma': [0.01],
        'ctx_len': [100],
    },
    'gamma': {
        'n_topics': [20],
        'gamma': [0.005, 0.05, 0.1, 0.5],
        'ctx_len': [100],
    },
    'n_topics_gamma': {
        'n_topics': [10, 20, 30, 50, 70, 100],
        'gamma': [0.005, 0.05, 0.01, 0.1, 0.5],
        'ctx_len': [100],
    },
    'ctx_len': {
        'n_topics': [20],
        'gamma': [0.01],
        'ctx_len': [25, 50, 75],
    }
}

exp_config = experiments[exp_mode]

for n_topics in exp_config['n_topics']:
    for gamma in exp_config['gamma']:
        for ctx_len in exp_config['ctx_len']:

            perplexity = PerplexityMetric(tag='perplexity')
            train_bow = build_bow(train_tokens, train_bounds, vocab_size)
            assert train_bow.sum() == len(train_tokens)
            npmi_coherence = NPMICoherenceMetric(bow=train_bow, top_k=10, tag='coherence')
            stratified_perplexity = StratifiedPerplexityMetric(
                tag='stratified_perplexity', n_topics=n_topics
            )
            topic_uniqueness = TopicUniquenessMetric(top_k=10, tag='uniqueness')

            metrics = {
                'perplexity': perplexity.history,
                'npmi_coherence': npmi_coherence.history,
                'stratified_perplexity': stratified_perplexity.history,
                'stratified_pt_perplexity': stratified_perplexity.per_topic_history,
                # Среднее U_t по темам (macro-uniqueness) по эпохам.
                'topic_uniqueness': topic_uniqueness.history,
                # Значения U_t по отдельным темам по эпохам.
                'topic_pt_uniqueness': topic_uniqueness.per_topic_history,
            }

            # Вариант модели выбирается через model_cls.
            model = model_cls(
                vocab_size=vocab_size,
                ctx_len=ctx_len,
                n_topics=n_topics,
                gamma=gamma,
                metrics=[
                    perplexity,
                    npmi_coherence,
                    stratified_perplexity,
                    topic_uniqueness,
                ],
                regularizers=[],
            )

            print(
                'variant_{variant}_n_topics_{n_topics}_gamma_{gamma}'
                '_ctx_len_{ctx_len}_{structured_data_prefix}'.format(
                    variant=model_variant, n_topics=n_topics, gamma=gamma,
                    ctx_len=ctx_len, structured_data_prefix=structured_data_prefix
                )
            )

            model.fit_with_test(
                train_batches=train_loader,
                test_batches=test_loader,
                max_iter=50,
                verbose=2,
                seed=42,
                num_batches_before_update=1,
                save_hist=False,
            )

            # Метка варианта модели добавлена в имя файла, чтобы результаты
            # полной и абляционной моделей не перезаписывали друг друга.
            result_filename = (
                'result_n_topics_{n_topics}_gamma_{gamma}_ctx_len_{ctx_len}'
                '_{structured_data_prefix}_{variant}.json'.format(
                    n_topics=n_topics, gamma=gamma, ctx_len=ctx_len,
                    structured_data_prefix=structured_data_prefix,
                    variant=model_variant,
                )
            )
            result_filepath = RESULT_PATH + result_filename

            with open(result_filepath, "w") as f:
                json.dump(metrics, f, indent=4)
