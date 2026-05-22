"""
run_main_table_imbalanced.py — сравнительная таблица базовых моделей
на коллекциях 20 Newsgroups с искусственным тематическим дисбалансом.

Назначение
----------
Скрипт закрывает сравнительную часть гипотезы Г1 ВКР: обучает базовые
модели (LDA, NMF, BERTopic, CombinedTM), а также полную и абляционную
AARTM на тех же четырёх коллекциях, что и экспериментальный стенд
ВКР, и сводит результаты в таблицу. Промежуточный вывод Г1 (слабая
абсолютная деградация AARTM) уже получен экспериментами 1--3; данный
скрипт даёт прямое сравнение AARTM с базовыми методами, без которого
вывод о её преимуществе на редких темах сделать нельзя.

Чем отличается от репозиторного run_main_table.py
-------------------------------------------------
Репозиторный run_main_table.py вызывает prepare_data, который грузит
СБАЛАНСИРОВАННЫЙ корпус 20 Newsgroups из sklearn и строит собственные
словарь и счётчики. Для ВКР это не подходит по двум причинам:

  1. Нужны не балансированные данные, а четыре конкретные коллекции
     ВКР: full, minus_rel, minus_med, imbalanced_test. Они порождаются
     скриптом dataset_preparation.py и сохранены как структурированные
     pickle-файлы (data/structured/<prefix>_structured_data.pkl) вместе
     с общим словарём data/corpuses/<corpus>_corpus_vocab.json.

  2. Все базовые модели и функции оценки репозитория принимают объект
     PreparedData с полями train_bow, train_tfidf, train_tokens, vocab,
     y_train и т.д. Структурированный pickle ВКР хранит только массивы
     токенов и границы документов, без счётчиков, текстов и меток.

Поэтому данный скрипт НЕ вызывает prepare_data. Вместо него функция
build_prepared_data_from_vkr собирает корректный объект PreparedData
из артефактов ВКР:

  - токены/границы train и test берутся напрямую из структурированного
    pickle коллекции;
  - словарь берётся из corpus_vocab.json (общий для всех коллекций);
  - счётчики BOW и TF-IDF строятся теми же функциями build_bow и
    TfidfTransformer, что и в prepare_data;
  - метки классов y_train/y_test ВОССТАНАВЛИВАЮТСЯ из словаря
    train_result_structure, сохранённого в том же pickle: коллекция
    строится get_structured_data по категориям подряд (order_seed=-1),
    поэтому достаточно повторить counts[cat] меток для каждой категории;
  - сырые тексты, нужные BERTopic и CombinedTM, восстанавливаются
    повторным применением логики get_structured_data к исходному
    корпусу 20 Newsgroups: тот же отбор индексов по категориям с тем
    же order_seed=-1 даёт тексты документов в том же порядке.

После сборки PreparedData скрипт переиспользует механизм build_specs
из run_main_table.py. Спецификации LDA, NMF, BERTopic и CombinedTM
остаются репозиторными без изменений; спецификации полной и
абляционной AARTM заменяются на версионно-совместимые (см. ниже).

Совместимость с актуальной версией cartm
----------------------------------------
Репозиторный код experiments/ местами рассинхронизирован с актуальной
версией пакета cartm. Затронуты три места, все относящиеся к AARTM:

  1. Функция обучения fit_topic_model вызывает model.fit(data=...,
     ctx_bounds=...). Текущий ModelBase.fit принимает батчи первым
     позиционным аргументом без параметров data/ctx_bounds, поэтому
     прежний вызов даёт TypeError: unexpected keyword argument 'data'.

  2. Функция aartm_phi_pwt вызывает model.renormalize_phi(batch=...,
     phi=...) и распаковывает два значения. Текущий renormalize_phi
     имеет сигнатуру renormalize_phi(p_w, phi) и возвращает одну
     матрицу, поэтому прежний вызов даёт TypeError: unexpected keyword
     argument 'batch'. От aartm_phi_pwt зависят и evaluate_aartm
     (eval_fn), и aartm_topic_words (topic_words_fn).

  3. Абляционный класс AttentiveTopicModelNoNWT переопределяет _step
     по устаревшей сигнатуре (6 возвращаемых значений, аргумент
     grad_reg) и несовместим с текущим методом обучения.

Поэтому для AARTM скрипт не использует репозиторные fit_topic_model,
evaluate_aartm, aartm_topic_words и класс AttentiveTopicModelNoNWT.
Вместо них применяются версионно-совместимые локальные аналоги:
fit_aartm_compatible (обучение через model.fit с корректной
позиционной сигнатурой), aartm_phi_pwt_compatible (вызов
renormalize_phi(p_w, phi)), aartm_eval_compatible,
aartm_topic_words_compatible и абляционный класс
AttentiveTopicModelAblNoNWT с совместимым _step. Все они повторяют
способ работы с моделью из рабочих стендов ВКР exp_stand.py и
exp_stand_abl.py. На LDA, NMF, BERTopic и CombinedTM эта
несовместимость не распространяется~--- их спецификации из build_specs
используются без изменений.

Восстановление меток и текстов: обоснование
-------------------------------------------
get_structured_data при order_seed=-1 (значение, которое использует
dataset_preparation.py) детерминирован: для каждой категории cat он
берёт первые test_structure[cat] документов в тест, следующие
dataset_structure[cat] — в обучение, затем индексы сортируются. Тот же
проход по unique-категориям с тем же order_seed, применённый к меткам
и текстам исходного 20 Newsgroups, даёт y_train/y_test и тексты,
порядок которых совпадает с порядком документов в pickle. Совпадение
проверяется ассертом: число восстановленных меток обязано равняться
числу документов в pickle (по сумме train_result_structure).

Запуск
------
    python run_main_table_imbalanced.py --dataset_prefix minus_rel \\
        --models lda,nmf,bertopic,ctm,aartm,aartm_no_nwt --seeds 0,1,2

    --dataset_prefix : full | minus_rel | minus_med | imbalanced_test
    --corpus_prefix  : префикс файла словаря (по умолчанию 'full')
    --structured_dir : каталог структурированных pickle (data/structured)
    --corpus_dir     : каталог словаря (data/corpuses)
    --out_dir        : каталог результатов (results/main_table_imbalanced)
    остальные аргументы (n_topics, gamma, ctx_len, max_iter, seeds, ...)
    совпадают с run_main_table.py.

Для закрытия Г1 достаточно прогнать три несбалансированные коллекции
и для контроля — сбалансированную full:
    for ds in full minus_rel minus_med imbalanced_test; do
      python run_main_table_imbalanced.py --dataset_prefix $ds
    done
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer

from cartm.preprocessing import CorpusLoader, BatchedCorpusLoader, build_bow
from cartm import AttentiveTopicModel
from cartm.core import EPSILON, norm, calc_attn

# Структура PreparedData, агрегирование, построитель регуляризаторов
# и функции оценки переиспользуются из репозиторного кода.
from experiments.common import (
    PreparedData,
    aggregate_results,
    build_regularizers,
    npmi_score,
    topic_diversity,
    topic_sparsity,
    mean_nearest_hellinger,
    classification_scores,
    infer_doc_topics_aartm,
)
from experiments.run_main_table import (
    build_specs,
    parse_csv_list,
    ModelSpec,
)
from experiments.topic_eval import phi_to_topic_words, save_topic_words_list


# ----------------------------------------------------------------------
# Абляционный вариант AARTM без члена N_wt, совместимый с актуальной
# версией cartm.
#
# Репозиторный класс AttentiveTopicModelNoNWT (experiments/model_no_N_wt.py)
# переопределяет _step и _update_phi по СТАРОЙ сигнатуре: _step возвращает
# 6 значений и принимает аргумент grad_reg. Текущая версия cartm
# использует _step с 4 возвращаемыми значениями без grad_reg, поэтому
# репозиторный абляционный класс несовместим с методом обучения fit и
# вызывает ошибку.
#
# Здесь определён абляционный класс с СОВМЕСТИМОЙ сигнатурой: он
# переопределяет только _step, сохраняя интерфейс базового
# AttentiveTopicModel и обнуляя счётчик N_wt. Класс идентичен
# AttentiveTopicModelAblNoNWT из стенда ВКР exp_stand_abl.py.
# ----------------------------------------------------------------------
class AttentiveTopicModelAblNoNWT(AttentiveTopicModel):
    """AARTM без контекстного члена N_wt (абляция, совместимая с fit)."""

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


# Конфигурация коллекций ВКР. Полностью совпадает с config_structures
# из dataset_preparation.py и приводится здесь, чтобы скрипт мог
# восстановить метки и тексты без импорта этого модуля.
CATEGORIES_QTY = 20
CATEGORY_MED = 13  # sci.med
CATEGORY_REL = 15  # soc.religion.christian

CONFIG_STRUCTURES = {
    'full': {
        'dataset_structure': {i: 1000 for i in range(CATEGORIES_QTY)},
        'test_structure': {i: 50 for i in range(CATEGORIES_QTY)},
    },
    'imbalanced_test': {
        'dataset_structure': {i: 1000 for i in range(CATEGORIES_QTY)},
        'test_structure': {i: (10 if i == CATEGORY_REL else 50)
                           for i in range(CATEGORIES_QTY)},
    },
    'minus_rel': {
        'dataset_structure': {i: (50 if i == CATEGORY_REL else 1000)
                              for i in range(CATEGORIES_QTY)},
        'test_structure': {i: (10 if i == CATEGORY_REL else 50)
                           for i in range(CATEGORIES_QTY)},
    },
    'minus_med': {
        'dataset_structure': {i: (50 if i == CATEGORY_MED else 1000)
                              for i in range(CATEGORIES_QTY)},
        'test_structure': {i: (10 if i == CATEGORY_MED else 50)
                           for i in range(CATEGORIES_QTY)},
    },
}

STRUCTURED_DATA_FILENAME = 'structured_data.pkl'
VOCAB_FILENAME = 'corpus_vocab.json'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Сравнительная таблица базовых моделей на '
                    'несбалансированных коллекциях 20NG.'
    )
    parser.add_argument('--dataset_prefix', type=str, default='minus_rel',
                        choices=list(CONFIG_STRUCTURES),
                        help='коллекция ВКР')
    parser.add_argument('--corpus_prefix', type=str, default='full',
                        help='префикс файла словаря corpus_vocab.json')
    parser.add_argument('--structured_dir', type=str, default='data/structured',
                        help='каталог структурированных pickle-файлов')
    parser.add_argument('--corpus_dir', type=str, default='data/corpuses',
                        help='каталог файла словаря')
    parser.add_argument('--out_dir', type=str,
                        default='results/main_table_imbalanced')
    parser.add_argument('--models', type=str,
                        default='aartm,aartm_no_nwt,lda,nmf,bertopic,ctm')
    parser.add_argument('--n_topics', type=int, default=20)
    parser.add_argument('--ctx_len', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--self_aware_context', action='store_true')
    parser.add_argument('--num_attn_passes', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--decorrelation_tau', type=float, default=0.0)
    parser.add_argument('--seeds', type=str, default='0,1,2')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2')
    return parser.parse_args()


def doc_count_from_structure(result_structure: dict) -> int:
    """Полное число документов по словарю {категория: число}."""
    return int(sum(result_structure.values()))


def recover_labels_and_texts(result_structure, raw_targets, raw_texts,
                             dataset_structure, test_structure, is_train):
    """
    Восстанавливает метки классов и сырые тексты для одной части
    (train или test) коллекции ВКР.

    Повторяет отбор индексов из cartm.preprocessing.get_structured_data
    при order_seed=-1: проход по категориям в порядке np.unique, для
    каждой категории первые test_structure[cat] документов идут в тест,
    следующие dataset_structure[cat] — в обучение; затем индексы
    сортируются. Тот же проход, применённый к меткам и текстам
    исходного 20 Newsgroups, даёт y и тексты в том же порядке, что и
    документы в структурированном pickle.

    Параметр result_structure (train_ или test_result_structure из
    pickle) задаёт фактическое число взятых документов по категориям
    и служит для проверки.
    """
    unique_cats = np.unique(raw_targets)
    selected_indices = []

    for cat in unique_cats:
        cat = int(cat)
        cat_idx = np.where(raw_targets == cat)[0]
        n_available = len(cat_idx)

        # Тестовая часть категории идёт первой.
        n_test_req = test_structure.get(cat, 0)
        n_test_take = min(n_test_req, n_available)
        test_idx = cat_idx[:n_test_take]
        remaining_idx = cat_idx[n_test_take:]

        # Обучающая часть — из остатка категории.
        n_train_req = dataset_structure.get(cat, len(remaining_idx))
        n_train_take = min(n_train_req, len(remaining_idx))
        train_idx = remaining_idx[:n_train_take]

        selected_indices.append(train_idx if is_train else test_idx)

    selected_indices = np.concatenate(selected_indices) if selected_indices \
        else np.array([], dtype=int)
    # Финальная сортировка — как в get_structured_data при order_seed=-1.
    selected_indices = np.sort(selected_indices)

    y = raw_targets[selected_indices]
    texts = [raw_texts[i] for i in selected_indices]

    # Проверка согласованности с pickle: число восстановленных
    # документов обязано совпасть с суммой result_structure.
    expected = doc_count_from_structure(result_structure)
    assert len(y) == expected, (
        f"Восстановлено {len(y)} документов, в pickle ожидается {expected}. "
        f"Проверьте, что dataset_preparation.py и данный скрипт используют "
        f"одинаковую конфигурацию коллекций."
    )
    return y, texts


def build_prepared_data_from_vkr(args) -> PreparedData:
    """
    Собирает объект PreparedData из артефактов ВКР для одной коллекции.

    Источники: структурированный pickle коллекции (токены, границы,
    result_structure), общий словарь corpus_vocab.json и исходный
    корпус 20 Newsgroups (для восстановления меток и текстов).
    """
    if args.dataset_prefix not in CONFIG_STRUCTURES:
        raise ValueError(f"Неизвестная коллекция: {args.dataset_prefix}")

    structured_path = (Path(args.structured_dir)
                       / f"{args.dataset_prefix}_{STRUCTURED_DATA_FILENAME}")
    vocab_path = (Path(args.corpus_dir)
                  / f"{args.corpus_prefix}_{VOCAB_FILENAME}")

    if not structured_path.exists():
        raise FileNotFoundError(
            f"Не найден структурированный pickle: {structured_path}. "
            f"Сгенерируйте его скриптом dataset_preparation.py."
        )
    if not vocab_path.exists():
        raise FileNotFoundError(f"Не найден словарь: {vocab_path}.")

    # --- Структурированные токены коллекции ---------------------------
    with open(structured_path, 'rb') as f:
        structured_data = pickle.load(f)
    (train_tokens, train_bounds, train_result_structure), \
        (test_tokens, test_bounds, test_result_structure) = structured_data

    train_tokens = jnp.asarray(train_tokens)
    train_bounds = jnp.asarray(train_bounds)
    test_tokens = jnp.asarray(test_tokens)
    test_bounds = jnp.asarray(test_bounds)

    # --- Словарь ------------------------------------------------------
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    id2word = {idx: word for word, idx in vocab.items()}

    # --- Счётчики BOW и TF-IDF (как в prepare_data) -------------------
    # build_bow уже возвращает scipy.sparse.csr_matrix.
    train_bow = build_bow(train_tokens, train_bounds, vocab_size)
    test_bow = build_bow(test_tokens, test_bounds, vocab_size)

    tfidf = TfidfTransformer(norm='l2')
    train_tfidf = tfidf.fit_transform(train_bow)
    test_tfidf = tfidf.transform(test_bow)

    # --- Метки и тексты из исходного 20 Newsgroups --------------------
    full_20ng = fetch_20newsgroups(
        data_home='./data/', subset='all',
        remove=('headers', 'footers', 'quotes'),
    )
    raw_texts = full_20ng.data
    raw_targets = np.asarray(full_20ng.target)

    cfg = CONFIG_STRUCTURES[args.dataset_prefix]
    y_train, train_texts = recover_labels_and_texts(
        train_result_structure, raw_targets, raw_texts,
        cfg['dataset_structure'], cfg['test_structure'], is_train=True,
    )
    y_test, test_texts = recover_labels_and_texts(
        test_result_structure, raw_targets, raw_texts,
        cfg['dataset_structure'], cfg['test_structure'], is_train=False,
    )

    # CorpusLoader со словарём — нужен спецификациям BERTopic/CTM,
    # которым требуется поле loader для токенизации текстов.
    loader = CorpusLoader(
        lower=True, min_token_len=3, max_token_len=20,
        vocabulary=vocab,
    )

    print('=== Несбалансированная коллекция ВКР ===')
    print(f"Коллекция:        {args.dataset_prefix}")
    print(f"Документов train: {len(train_texts)}")
    print(f"Документов test:  {len(test_texts)}")
    print(f"Размер словаря:   {vocab_size}")
    print(f"Токенов train:    {len(train_tokens)}")
    print(f"Токенов test:     {len(test_tokens)}")
    print(f"train_result_structure: {train_result_structure}")
    print(f"test_result_structure:  {test_result_structure}")

    return PreparedData(
        dataset_name=args.dataset_prefix,
        train_texts=train_texts,
        test_texts=test_texts,
        train_texts_filtered=train_texts,
        test_texts_filtered=test_texts,
        y_train=y_train,
        y_test=y_test,
        loader=loader,
        train_tokens=train_tokens,
        train_bounds=train_bounds,
        test_tokens=test_tokens,
        test_bounds=test_bounds,
        train_bow=train_bow,
        test_bow=test_bow,
        train_tfidf=train_tfidf,
        test_tfidf=test_tfidf,
        vocab=vocab,
        id2word=id2word,
    )


def fit_aartm_compatible(model_cls, data: PreparedData, args, seed: int):
    """
    Обучает вариант AARTM (полный или абляционный) с сигнатурой,
    совместимой с текущей версией пакета cartm.

    Зачем нужна эта функция. Репозиторная fit_local_model вызывает
    experiments.common.fit_topic_model, который обращается к
    model.fit(data=..., ctx_bounds=...). В актуальной версии cartm
    метод ModelBase.fit принимает батчи ПЕРВЫМ позиционным аргументом
    и не имеет параметров data/ctx_bounds, поэтому прежний вызов падает
    с TypeError: unexpected keyword argument 'data'. Здесь обучение
    выполняется напрямую через model.fit(batches, ...) тем же способом,
    что и в рабочем стенде ВКР exp_stand.py.

    Возвращает кортеж (model, elapsed, cache) в формате, который
    ожидают ModelSpec.eval_fn и ModelSpec.topic_words_fn.
    """
    from time import perf_counter

    regs = build_regularizers(args.decorrelation_tau, "tw")
    model = model_cls(
        vocab_size=len(data.vocab),
        ctx_len=args.ctx_len,
        n_topics=args.n_topics,
        gamma=args.gamma,
        self_aware_context=args.self_aware_context,
        regularizers=regs,
    )

    train_batches = BatchedCorpusLoader(
        data.train_tokens, data.train_bounds, batch_size=args.batch_size
    )

    t0 = perf_counter()
    # Батчи передаются ПЕРВЫМ позиционным аргументом — совместимо
    # с актуальной сигнатурой ModelBase.fit.
    model.fit(
        train_batches,
        num_attn_passes=args.num_attn_passes,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=0,
        seed=seed,
    )
    elapsed = perf_counter() - t0
    return model, elapsed, {}


def aartm_phi_pwt_compatible(model) -> np.ndarray:
    """
    Возвращает матрицу p(w|t) обученной модели AARTM.

    Зачем нужна эта функция. Репозиторная aartm_phi_pwt в
    experiments/common.py вызывает model.renormalize_phi(batch=...,
    phi=...) и распаковывает два значения. В актуальной версии cartm
    метод renormalize_phi имеет сигнатуру renormalize_phi(p_w, phi) и
    возвращает одну матрицу, из-за чего прежний вызов завершается
    ошибкой TypeError: unexpected keyword argument 'batch'.

    Здесь используется корректный вызов renormalize_phi(p_w=model.p_w,
    phi=model.phi)~--- тот же, что и в стенде ВКР exp_stand_abl.py.
    Матрица model.phi хранит p(t|w); renormalize_phi преобразует её в
    p(w|t), как и при расчёте метрик внутри модели.
    """
    phi_wt = model.renormalize_phi(p_w=model.p_w, phi=model.phi)
    return np.asarray(jax.device_get(phi_wt))


def aartm_eval_compatible(model, data: PreparedData, args, *, cache=None,
                          seed: int = 0, **kwargs) -> dict:
    """
    Версионно-совместимая оценка AARTM. Повторяет логику репозиторной
    evaluate_aartm, но получает p(w|t) через aartm_phi_pwt_compatible.

    Метрики: NPMI, разнообразие тем, разреженность, среднее
    хеллингерово расстояние до ближайшей темы, а также точность и
    macro-F1 классификатора на распределениях тем документов.
    """
    phi_wt = aartm_phi_pwt_compatible(model)

    batches_train = BatchedCorpusLoader(
        data.train_tokens, data.train_bounds, batch_size=args.batch_size
    )
    X_train = infer_doc_topics_aartm(
        model, batches_train, num_attn_passes=args.num_attn_passes
    )
    batches_test = BatchedCorpusLoader(
        data.test_tokens, data.test_bounds, batch_size=args.batch_size
    )
    X_test = infer_doc_topics_aartm(
        model, batches_test, num_attn_passes=args.num_attn_passes
    )

    metrics = {
        "npmi_10": npmi_score(phi_wt, data.train_bow, top_k=10),
        "topic_diversity_25": topic_diversity(phi_wt, top_k=25),
        "topic_sparsity": topic_sparsity(phi_wt),
        "topic_hellinger": mean_nearest_hellinger(phi_wt),
    }
    metrics.update(
        classification_scores(X_train, data.y_train, X_test, data.y_test,
                              seed=seed)
    )
    return metrics


def aartm_topic_words_compatible(model, data: PreparedData, cache,
                                 top_k: int) -> list[list[str]]:
    """
    Версионно-совместимое извлечение топ-слов тем AARTM. Заменяет
    репозиторную aartm_topic_words, опирающуюся на устаревшую
    aartm_phi_pwt.
    """
    phi_wt = aartm_phi_pwt_compatible(model)
    return phi_to_topic_words(phi_wt, data.id2word, top_k=top_k)


def make_aartm_eval(args):
    """Возвращает eval_fn для AARTM с привязанным объектом args."""
    def _eval(model, data, *, cache=None, seed=0, **kwargs):
        return aartm_eval_compatible(model, data, args,
                                     cache=cache, seed=seed)
    return _eval


def patch_aartm_specs(specs: dict, args) -> dict:
    """
    Заменяет fit_fn, eval_fn и topic_words_fn у спецификаций AARTM на
    версионно-совместимые. Репозиторные eval_fn (evaluate_aartm) и
    topic_words_fn (aartm_topic_words) опираются на устаревшую
    aartm_phi_pwt, вызывающую renormalize_phi с несуществующим
    аргументом batch; поэтому они также заменяются. Спецификации LDA,
    NMF, BERTopic и CombinedTM остаются репозиторными без изменений.
    """
    aartm_eval = make_aartm_eval(args)
    if "aartm" in specs:
        specs["aartm"] = ModelSpec(
            name=specs["aartm"].name,
            fit_fn=lambda data, seed: fit_aartm_compatible(
                AttentiveTopicModel, data, args, seed),
            eval_fn=aartm_eval,
            topic_words_fn=aartm_topic_words_compatible,
        )
    if "aartm_no_nwt" in specs:
        specs["aartm_no_nwt"] = ModelSpec(
            name=specs["aartm_no_nwt"].name,
            fit_fn=lambda data, seed: fit_aartm_compatible(
                AttentiveTopicModelAblNoNWT, data, args, seed),
            eval_fn=aartm_eval,
            topic_words_fn=aartm_topic_words_compatible,
        )
    return specs


if __name__ == '__main__':
    args = parse_args()

    out_dir = Path(args.out_dir) / args.dataset_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    nltk.download('stopwords')

    # Сборка PreparedData из артефактов ВКР вместо вызова prepare_data.
    data = build_prepared_data_from_vkr(args)

    selected = parse_csv_list(args.models)
    # build_specs репозиторный; затем спецификации AARTM заменяются
    # на версионно-совместимый fit (см. patch_aartm_specs).
    specs = build_specs(args)
    specs = patch_aartm_specs(specs, args)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]

    rows = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        for key in selected:
            if key not in specs:
                print(f"Пропуск '{key}': нет такой спецификации модели")
                continue
            spec = specs[key]
            print(f"Обучение {spec.name} ...")

            try:
                model, elapsed, cache = spec.fit_fn(data, seed)
            except ImportError as e:
                # BERTopic и CombinedTM требуют дополнительных зависимостей;
                # при их отсутствии модель пропускается, остальные считаются.
                print(f"Пропуск {spec.name}: отсутствует зависимость: {e}")
                continue

            metrics = spec.eval_fn(model, data, cache=cache, seed=seed)
            metrics.update({
                'dataset': args.dataset_prefix,
                'model': spec.name,
                'seed': seed,
                'n_topics': args.n_topics,
                'train_time_sec': elapsed,
            })
            rows.append(metrics)

            topic_words = spec.topic_words_fn(model, data, cache, 15)
            save_topic_words_list(
                topic_words,
                out_dir / f"top_words_{spec.name}_seed{seed}.txt",
            )

    if not rows:
        print("\nНи одна модель не обучена — таблица не построена.")
        raise SystemExit(1)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_dir / 'main_table_raw.csv', index=False)

    summary_df = aggregate_results(raw_df, group_cols=['dataset', 'model'])
    summary_df.to_csv(out_dir / 'main_table_summary.csv', index=False)

    print()
    print(raw_df.to_string(index=False))
    print()
    print(summary_df.to_string(index=False))
    print(f"\nСохранено в {out_dir}")
