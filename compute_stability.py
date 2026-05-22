"""
compute_stability.py — расчёт устойчивости тем модели AARTM между
прогонами с разными зёрнами инициализации.

Назначение
----------
Скрипт постобработки. Он не обучает модели, а считает меру устойчивости
по файлам топ-слов topwords_*.json, которые порождает exp_stand_abl.py.
Разделение обучения и расчёта устойчивости сделано сознательно: обучение
на JAX дорогое, а сравнение наборов топ-слов — дешёвое, и его удобно
пересчитывать отдельно (в том числе при разных top_k).

Метод
-----
Устойчивость измеряется по образцу experiments/run_stability.py:
для каждой пары прогонов с разными зёрнами темы одной модели оптимально
сопоставляются темам другой (венгерский алгоритм, scipy linear_sum_assignment),
а близость сопоставленных тем — средняя мера Жаккара между их наборами
топ-слов. Итоговая устойчивость конфигурации — среднее по всем парам зёрен.

Отличие от run_stability.py репозитория
---------------------------------------
Репозиторный run_stability.py работает в рамках фреймворка experiments/
(HuggingFace-датасеты, build_specs, обучение моделей внутри скрипта).
Стенд ВКР (exp_stand_abl.py) идёт другим путём: коллекции 20NG с
искусственным дисбалансом, обучение пакетом cartm напрямую. Поэтому
здесь функция matched_topic_jaccard и агрегирование воспроизведены
локально (идентичны репозиторным по логике), а входом служат не модели
в памяти, а уже сохранённые файлы топ-слов. Зависимость от тяжёлого
модуля experiments.common (sklearn, datasets) при этом не нужна.

Вход
----
Каталог с файлами topwords_*.json. Каждый файл описывает один прогон:
поля dataset, variant, seed, n_topics, gamma, ctx_len, top_k, topics.
Прогоны группируются по конфигурации (dataset, variant, n_topics, gamma,
ctx_len); устойчивость считается внутри каждой группы по парам зёрен.

Запуск
------
    python compute_stability.py [--results_dir DIR] [--out_dir DIR]
                                [--top_k K] [--dataset NAME] [--variant NAME]

    --results_dir : каталог с файлами topwords_*.json (default: data/results)
    --out_dir     : каталог для CSV-сводок (default: data/results/stability)
    --top_k       : размер набора топ-слов для сравнения (default: 10)
    --dataset     : ограничить расчёт одной коллекцией (необязательно)
    --variant     : ограничить расчёт одним вариантом модели (необязательно)

Пример (после серии прогонов exp_stand_abl.py с зёрнами 0..4):
    python compute_stability.py --results_dir data/results --top_k 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# Поля, однозначно задающие конфигурацию прогона (всё, кроме зерна).
CONFIG_KEYS = ('dataset', 'variant', 'n_topics', 'gamma', 'ctx_len')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Расчёт устойчивости тем AARTM по файлам топ-слов.'
    )
    parser.add_argument('--results_dir', type=str, default='data/results',
                        help='каталог с файлами topwords_*.json')
    parser.add_argument('--out_dir', type=str, default='data/results/stability',
                        help='каталог для CSV-сводок')
    parser.add_argument('--top_k', type=int, default=10,
                        help='размер набора топ-слов для сравнения')
    parser.add_argument('--dataset', type=str, default=None,
                        help='ограничить расчёт одной коллекцией')
    parser.add_argument('--variant', type=str, default=None,
                        help='ограничить расчёт одним вариантом модели')
    return parser.parse_args()


def matched_topic_jaccard(
    topic_words_a: list[list[str]],
    topic_words_b: list[list[str]],
    top_k: int = 10,
) -> float:
    """
    Средняя мера Жаккара между оптимально сопоставленными темами двух
    прогонов. Сопоставление — венгерский алгоритм на матрице (1 - сходство).

    Воспроизводит matched_topic_jaccard из experiments/run_stability.py.
    """
    A = [set(words[:top_k]) for words in topic_words_a]
    B = [set(words[:top_k]) for words in topic_words_b]

    if len(A) == 0 or len(B) == 0:
        return float('nan')

    sim = np.zeros((len(A), len(B)), dtype=np.float32)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            union = len(a | b)
            sim[i, j] = 0.0 if union == 0 else len(a & b) / union

    row_ind, col_ind = linear_sum_assignment(1.0 - sim)
    return float(sim[row_ind, col_ind].mean())


def aggregate_results(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Среднее и стандартное отклонение числовых метрик по группам.
    Воспроизводит aggregate_results из experiments/common.py.
    """
    metric_cols = [c for c in df.columns if c not in group_cols]
    agg = df.groupby(group_cols)[metric_cols].agg(['mean', 'std']).reset_index()
    agg.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    return agg


def load_topwords_files(results_dir: str) -> list[dict]:
    """Читает все файлы topwords_*.json из каталога результатов."""
    pattern = os.path.join(results_dir, 'topwords_*.json')
    paths = sorted(glob.glob(pattern))

    records = []
    for path in paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  пропущен файл {os.path.basename(path)}: {e}")
            continue

        missing = [k for k in (*CONFIG_KEYS, 'seed', 'topics') if k not in payload]
        if missing:
            print(f"  пропущен файл {os.path.basename(path)}: "
                  f"нет полей {missing}")
            continue

        payload['_path'] = path
        records.append(payload)

    return records


def config_key(record: dict) -> tuple:
    """Кортеж-идентификатор конфигурации прогона (без зерна)."""
    return tuple(record[k] for k in CONFIG_KEYS)


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = load_topwords_files(args.results_dir)
    if not records:
        print(f"В каталоге {args.results_dir} не найдено файлов topwords_*.json.")
        raise SystemExit(1)

    # Необязательная фильтрация по коллекции / варианту модели.
    if args.dataset is not None:
        records = [r for r in records if r['dataset'] == args.dataset]
    if args.variant is not None:
        records = [r for r in records if r['variant'] == args.variant]

    if not records:
        print("После фильтрации не осталось ни одного прогона.")
        raise SystemExit(1)

    # Группировка прогонов по конфигурации.
    groups: dict[tuple, dict[int, list]] = defaultdict(dict)
    for rec in records:
        key = config_key(rec)
        seed = rec['seed']
        if seed in groups[key]:
            print(f"  предупреждение: для конфигурации {key} зерно {seed} "
                  f"встречается повторно, использован последний файл")
        groups[key][seed] = rec['topics']

    rows = []

    for key, per_seed_topics in sorted(groups.items()):
        dataset, variant, n_topics, gamma, ctx_len = key
        seeds = sorted(per_seed_topics)

        if len(seeds) < 2:
            print(f"=== {key} === пропущена: только {len(seeds)} зерно "
                  f"(нужно не менее двух для оценки устойчивости)")
            continue

        print(f"=== dataset={dataset} variant={variant} "
              f"n_topics={n_topics} gamma={gamma} ctx_len={ctx_len} === "
              f"зёрна: {seeds}")

        for seed_a, seed_b in combinations(seeds, 2):
            score = matched_topic_jaccard(
                per_seed_topics[seed_a],
                per_seed_topics[seed_b],
                top_k=args.top_k,
            )
            rows.append({
                'dataset': dataset,
                'variant': variant,
                'n_topics': n_topics,
                'gamma': gamma,
                'ctx_len': ctx_len,
                'seed_a': seed_a,
                'seed_b': seed_b,
                f'matched_jaccard_top{args.top_k}': score,
            })

    if not rows:
        print("\nНи для одной конфигурации нет пары зёрен — устойчивость "
              "не вычислена. Запустите exp_stand_abl.py хотя бы с двумя "
              "разными значениями SEED.")
        raise SystemExit(1)

    raw_df = pd.DataFrame(rows)
    raw_path = os.path.join(args.out_dir, 'stability_raw.csv')
    raw_df.to_csv(raw_path, index=False)

    # Сводка: среднее и стандартное отклонение устойчивости по конфигурациям.
    # seed_a / seed_b исключаются из агрегирования как служебные столбцы.
    summary_input = raw_df.drop(columns=['seed_a', 'seed_b'])
    summary_df = aggregate_results(
        summary_input,
        group_cols=['dataset', 'variant', 'n_topics', 'gamma', 'ctx_len'],
    )
    summary_path = os.path.join(args.out_dir, 'stability_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print()
    print(raw_df.to_string(index=False))
    print()
    print(summary_df.to_string(index=False))
    print(f"\nСохранено:\n  {raw_path}\n  {summary_path}")
