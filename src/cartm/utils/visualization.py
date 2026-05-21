# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# Настройка стиля для отчётов
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def plot_per_topic_ppl(
    metric_history: List[Dict[str, Optional[float]]],
    topic_labels: Optional[List[str]] = None,
    title: str = "Per-topic Perplexity over Epochs",
    output_path: Optional[str] = None,
    highlight_low_frequency: Optional[List[int]] = None,
):
    """
    Визуализирует динамику per-topic PPL по эпохам.
    
    Args:
        metric_history: список dict'ов {topic_name: ppl_value} от get_per_topic_history()
        topic_labels: список читаемых названий тем (опционально)
        title: заголовок графика
        output_path: если указан, сохраняет график в файл
        highlight_low_frequency: индексы тем, которые известны как редкие (подсветка)
    """
    if not metric_history:
        print("⚠️ История метрики пуста. Убедитесь, что метрика вызывалась во время обучения.")
        return

    # Преобразуем в DataFrame для удобства
    data = []
    for epoch, entry in enumerate(metric_history):
        for topic_name, ppl in entry.items():
            if ppl is not None:  # пропускаем темы без данных
                topic_id = int(topic_name.replace("topic_", ""))
                data.append({
                    "epoch": epoch + 1,
                    "topic_id": topic_id,
                    "topic_label": topic_labels[topic_id] if topic_labels else f"Topic {topic_id}",
                    "ppl": ppl,
                    "is_rare": topic_id in (highlight_low_frequency or [])
                })
    
    df = pd.DataFrame(data)
    
    # Основной график: линии для каждой темы
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    # Разделяем обычные и редкие темы для визуального акцента
    common_df = df[~df["is_rare"]]
    rare_df = df[df["is_rare"]]
    
    # Редкие темы: жирные линии с маркерами
    for topic_id, group in rare_df.groupby("topic_id"):
        ax.plot(
            group["epoch"], group["ppl"],
            label=f"{group['topic_label'].iloc[0]} (rare)",
            marker="o", linewidth=2.5, markersize=4
        )
    
    # Обычные темы: тонкие линии без маркеров
    for topic_id, group in common_df.groupby("topic_id"):
        ax.plot(
            group["epoch"], group["ppl"],
            label=f"{group['topic_label'].iloc[0]}",
            linewidth=1, alpha=0.7
        )
    
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Per-topic Perplexity", fontsize=11)
    ax.set_title(title, fontsize=13, pad=15)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Легенда с переносом, если тем много
    if len(df["topic_label"].unique()) <= 15:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    else:
        ax.legend(ncol=2, fontsize=8)
    
    # Линия макро-среднего для ориентира
    macro_avg = df.groupby("epoch")["ppl"].mean()
    ax.plot(
        macro_avg.index, macro_avg.values,
        color="black", linestyle="--", linewidth=2,
        label="Macro-average (stratified)"
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ График сохранён: {output_path}")
    
    plt.show()
    return fig, ax


def plot_per_topic_ppl_boxplot(
    final_epoch_data: Dict[str, Optional[float]],
    topic_labels: Optional[List[str]] = None,
    title: str = "Per-topic Perplexity (Final Epoch)",
    output_path: Optional[str] = None,
):
    """
    Boxplot для per-topic PPL на финальной эпохе — компактный вид для отчёта.
    """
    topics = []
    values = []
    colors = []
    
    for topic_name, ppl in final_epoch_data.items():
        if ppl is not None:
            topic_id = int(topic_name.replace("topic_", ""))
            label = topic_labels[topic_id] if topic_labels else f"T{topic_id}"
            topics.append(label)
            values.append(ppl)
            # Цвет: зелёный для низкого PPL, красный для высокого
            colors.append("#2ecc71" if ppl < 100 else "#e74c3c" if ppl > 500 else "#f39c12")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    bars = ax.barh(topics, values, color=colors, edgecolor="black", alpha=0.9)
    
    ax.set_xlabel("Per-topic Perplexity", fontsize=11)
    ax.set_title(title, fontsize=13, pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    # Подписи значений на барах
    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
               f"{val:.0f}", va="center", fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ График сохранён: {output_path}")
    
    plt.show()
    return fig, ax