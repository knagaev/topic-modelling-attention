"""
exp_stand_abl.py — экспериментальный стенд для абляционного исследования
члена N_wt в модели AARTM и для исследования устойчивости по зёрнам.

Назначение
----------
Скрипт повторяет логику exp_stand.py, но добавляет три возможности:

    1. Выбор варианта модели:
        - 'full'   : полная модель AttentiveTopicModel (с членом N_wt в M-шаге);
        - 'no_nwt' : абляционный вариант без члена N_wt.
       Сравнение двух вариантов при идентичных прочих условиях обеспечивает
       проверку гипотезы Г2 (вклад контекстного члена N_wt в когерентность).

    2. Явное задание зерна инициализации (SEED). Зерно выносится в аргумент
       командной строки и передаётся в fit_with_test, а также записывается
       в имена выходных файлов. Это позволяет выполнить серию прогонов с
       разными зёрнами для оценки устойчивости тем (см. compute_stability.py).

    3. Сохранение топ-слов каждой темы. После обучения из матрицы Phi
       извлекаются топ-TOP_WORDS_QTY слов по каждой теме и сохраняются в
       отдельный JSON-файл topwords_*.json. Эти файлы — вход для скрипта
       compute_stability.py, который считает меру устойчивости тем между
       прогонами с разными зёрнами (сопоставление венгерским алгоритмом,
       мера Жаккара — по образцу experiments/run_stability.py).

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

ВАЖНО. Скрипт импортирует класс TopicUniquenessMetric из модуля
cartm.metrics.topic_uniqueness. Перед запуском поместите файл
topic_uniqueness.py в каталог пакета src/cartm/metrics/ рядом с
остальными метриками. Регистрировать класс в metrics/__init__.py
не обязательно — здесь используется прямой импорт из модуля.

Запуск
------
    python exp_stand_abl.py STRUCTURED_DATA_PREFIX EXP_MODE [MODEL_VARIANT] [SEED]

    STRUCTURED_DATA_PREFIX : ['full', 'imbalanced_test', 'minus_rel', 'minus_med']
    EXP_MODE               : ['base', 'gamma', 'n_topics_gamma']
    MODEL_VARIANT          : ['full', 'no_nwt']   (по умолчанию 'full')
    SEED                   : целое зерно инициализации (по умолчанию 42)

Имена выходных файлов содержат метку варианта модели и зерно, чтобы
прогоны разных вариантов и разных зёрен не перезаписывали друг друга:
    result_n_topics_{nt}_gamma_{g}_ctx_len_{cl}_{dataset}_{variant}_seed_{seed}.json
    topwords_n_topics_{nt}_gamma_{g}_ctx_len_{cl}_{dataset}_{variant}_seed_{seed}.json

Пример серии прогонов для оценки устойчивости (эталонная конфигурация,
пять зёрен, четыре коллекции):
    for ds in full minus_rel minus_med imbalanced_test; do
      for s in 0 1 2 3 4; do
        python exp_stand_abl.py $ds base full $s
      done
    done
"""

import pickle
import sys
import json

import jax
import jax.numpy as jnp
import numpy as np

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
# (см. инструкцию в шапке файла).
from cartm.metrics.topic_uniqueness import TopicUniquenessMetric
from cartm.regularization import DecorrelationRegularization


# Количество топ-слов, сохраняемых по каждой теме. Берётся с запасом
# (15), чтобы скрипт устойчивости мог сравнивать темы по top_k <= 15
# без повторного обучения; ср. experiments/run_stability.py, где
# topic_words_fn также сохраняет 15 слов, а сравнение идёт по top-10.
TOP_WORDS_QTY = 15


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
    print('Usage: python exp_stand_abl.py STRUCTURED_DATA_PREFIX EXP_MODE [MODEL_VARIANT] [SEED]')
    print("  STRUCTURED_DATA_PREFIX : ['full', 'imbalanced_test', 'minus_rel', 'minus_med']")
    print("  EXP_MODE               : ['base', 'gamma', 'n_topics_gamma']")
    print("  MODEL_VARIANT          : ['full', 'no_nwt']  (default: 'full')")
    print("  SEED                   : integer init seed (default: 42)")
    exit()

corpus_data_prefix = 'full'
structured_data_prefix = sys.argv[1]  # датасет
exp_mode = sys.argv[2]                # режим эксперимента
model_variant = sys.argv[3] if len(sys.argv) > 3 else 'full'  # вариант модели

if model_variant not in MODEL_VARIANTS:
    print(f"Unknown MODEL_VARIANT '{model_variant}'. Use one of: {list(MODEL_VARIANTS)}")
    exit()

# Зерно инициализации. Четвёртый позиционный аргумент; по умолчанию 42,
# что сохраняет совместимость с прежними прогонами стенда.
if len(sys.argv) > 4:
    try:
        seed = int(sys.argv[4])
    except ValueError:
        print(f"SEED must be an integer, got '{sys.argv[4]}'.")
        exit()
else:
    seed = 42

print(f"{structured_data_prefix=}")
print(f"{exp_mode=}")
print(f"{model_variant=}")
print(f"{seed=}")

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

# Обратный словарь индекс -> терм для извлечения топ-слов тем.
reverse_vocab = {idx: word for word, idx in vocab.items()}

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


def extract_top_words(model, reverse_vocab, top_k):
    """
    Извлекает топ-`top_k` слов каждой темы из обученной модели.

    Матрица model.phi хранит p(t|w). Для ранжирования слов внутри темы
    нужна матрица p(w|t); она получается тем же преобразованием
    renormalize_phi, что применяется при расчёте метрик NPMI и
    uniqueness (см. AttentiveTopicModel._calc_metrics_batch).

    Возвращает список длиной n_topics; каждый элемент — список из
    top_k термов, упорядоченных по убыванию p(w|t).
    """
    phi_wt = model.renormalize_phi(p_w=model.p_w, phi=model.phi)  # (W, T), p(w|t)
    phi_wt = np.asarray(phi_wt)

    # Индексы топ-слов по каждой теме, по убыванию вероятности.
    top_idx = np.argsort(-phi_wt, axis=0)[:top_k]  # (top_k, T)

    n_topics = phi_wt.shape[1]
    topics = []
    for t in range(n_topics):
        words = [reverse_vocab[int(top_idx[k, t])] for k in range(top_idx.shape[0])]
        topics.append(words)
    return topics


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

            # Вариант модели выбирается через model_cls
            # ('full' -> AttentiveTopicModel, 'no_nwt' -> AttentiveTopicModelAblNoNWT).
            model = model_cls(
                vocab_size=vocab_size,
                ctx_len=ctx_len,
                n_topics=n_topics,
                gamma=gamma,
                metrics=[perplexity, npmi_coherence, stratified_perplexity, topic_uniqueness],
                regularizers=[],
            )

            print(
                'variant_{variant}_seed_{seed}_n_topics_{n_topics}_gamma_{gamma}'
                '_ctx_len_{ctx_len}_{structured_data_prefix}'.format(
                    variant=model_variant, seed=seed, n_topics=n_topics, gamma=gamma,
                    ctx_len=ctx_len, structured_data_prefix=structured_data_prefix
                )
            )

            # Зерно инициализации передаётся явно: задаёт стартовую Phi
            # в _init_state и тем самым определяет конкретный прогон.
            model.fit_with_test(
                train_batches=train_loader,
                test_batches=test_loader,
                max_iter=50,
                verbose=2,
                seed=seed,
                num_batches_before_update=1,
                save_hist=False,
            )

            # --- Сохранение истории метрик ----------------------------
            # Метка варианта и зерно добавлены в имя файла, чтобы прогоны
            # разных вариантов и разных зёрен не перезаписывали друг друга.
            result_filename = (
                'result_n_topics_{n_topics}_gamma_{gamma}_ctx_len_{ctx_len}'
                '_{structured_data_prefix}_{variant}_seed_{seed}.json'.format(
                    n_topics=n_topics, gamma=gamma, ctx_len=ctx_len,
                    structured_data_prefix=structured_data_prefix,
                    variant=model_variant, seed=seed,
                )
            )
            result_filepath = RESULT_PATH + result_filename

            with open(result_filepath, "w") as f:
                json.dump(metrics, f, indent=4)

            # --- Сохранение топ-слов тем ------------------------------
            # Отдельный файл topwords_*.json — вход для compute_stability.py.
            # Содержит мета-информацию о прогоне и список топ-слов по темам.
            top_words = extract_top_words(model, reverse_vocab, TOP_WORDS_QTY)

            topwords_payload = {
                'dataset': structured_data_prefix,
                'variant': model_variant,
                'seed': seed,
                'n_topics': n_topics,
                'gamma': gamma,
                'ctx_len': ctx_len,
                'top_k': TOP_WORDS_QTY,
                'topics': top_words,
            }

            topwords_filename = (
                'topwords_n_topics_{n_topics}_gamma_{gamma}_ctx_len_{ctx_len}'
                '_{structured_data_prefix}_{variant}_seed_{seed}.json'.format(
                    n_topics=n_topics, gamma=gamma, ctx_len=ctx_len,
                    structured_data_prefix=structured_data_prefix,
                    variant=model_variant, seed=seed,
                )
            )
            topwords_filepath = RESULT_PATH + topwords_filename

            with open(topwords_filepath, "w") as f:
                json.dump(topwords_payload, f, indent=4, ensure_ascii=False)

            print(f"  saved metrics  -> {result_filename}")
            print(f"  saved topwords -> {topwords_filename}")
