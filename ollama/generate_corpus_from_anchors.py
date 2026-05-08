import json
import os
import time
import random
import requests
from tqdm import tqdm

# ================= 1. КОНФИГУРАЦИЯ =================
INPUT_FILE = "20ng_anchors_tfidf.json"
OUTPUT_FILE = "synthetic_20ng_anchors_llm.json"

# Настройки модели (оптимально для RTX 3060 Laptop 6GB)
MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
TEMPERATURE = 0.3
MAX_TOKENS = 200       # Ограничение длины вывода
RETRY_DELAY = 2        # Пауза при ошибке сервера

# Параметры генерации корпуса
NUM_DOCS_PER_TOPIC = 2  # Сколько документов генерировать для каждой темы
MIN_WORD_LEN = 3         # Фильтр коротких токенов в промпте

# ================= 2. ЗАГРУЗКА ЯКОРНЫХ ДАННЫХ =================
print(f"📂 Загрузка данных из {INPUT_FILE}...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Нормализация ключей (убираем пробелы в конце названий тем, если есть)
topics_data = {k.strip(): v for k, v in raw_data.items()}

print(f"📝 Найдено тем: {len(topics_data)}")
for topic, words in list(topics_data.items())[:3]:
    print(f"   Тема: {topic} | Примеры слов: {words[:3]}")

# ================= 3. ФУНКЦИЯ ГЕНЕРАЦИИ ЧЕРЕЗ OLLAMA =================
def generate_with_ollama(topic_name, anchor_words, max_retries=3):
    """
    Генерирует текст Usenet-поста, используя список якорных слов темы.
    """
    # Подготовка промпта: берем уникальные слова, сортируем для детерминизма
    unique_words = sorted(set(w for w in anchor_words if len(w) >= MIN_WORD_LEN))
    
    # Если слов слишком много, берем случайную выборку для разнообразия промптов
    if len(unique_words) > 30:
        selected_words = random.sample(unique_words, 30)
    else:
        selected_words = unique_words
        
    word_str = ", ".join(selected_words)
    
    system_instruction = (
        f"You are a user posting to the '{topic_name}' newsgroup in the 1990s. "
        f"Write a short, coherent forum post (80-150 words). "
        f"Naturally use some of these specific terms: {word_str}. "
        "Keep the tone technical, argumentative, or conversational as appropriate for the topic. "
        "Do not use markdown formatting."
    )

    payload = {
        "model": MODEL,
        "prompt": system_instruction,
        "temperature": TEMPERATURE,
        "num_predict": MAX_TOKENS,
        "stream": False
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            return result.get("response", "").strip()
        
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Ошибка запроса (попытка {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return None

# ================= 4. ОСНОВНОЙ ЦИКЛ ГЕНЕРАЦИИ =================
total_docs = NUM_DOCS_PER_TOPIC * len(topics_data)
print(f"\n🚀 Запуск генерации {total_docs} документов ({NUM_DOCS_PER_TOPIC} на тему)...")
print(f"💡 Убедитесь, что Ollama запущена: ollama serve")

generated_corpus = []
stats = {"success": 0, "fail": 0}

# Проходим по каждой теме
for topic_name, anchor_list in tqdm(topics_data.items(), desc="Темы"):
    # Для каждой темы генерируем N документов
    for i in range(NUM_DOCS_PER_TOPIC):
        text = generate_with_ollama(topic_name, anchor_list)
        
        if text:
            generated_corpus.append({
                "text": text,
                "true_topic": topic_name,
                "anchors_used_count": len(anchor_list)
            })
            stats["success"] += 1
        else:
            stats["fail"] += 1

# ================= 5. СОХРАНЕНИЕ И СТАТИСТИКА =================
print("\n📊 Статистика генерации:")
print(f"   Успешно: {stats['success']}")
print(f"   Ошибок: {stats['fail']}")

output_data = {
    "documents": generated_corpus,
    "model_used": MODEL,
    "parameters": {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "docs_per_topic": NUM_DOCS_PER_TOPIC,
        "input_source": INPUT_FILE
    },
    "stats": stats
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"💾 Результат сохранен в {OUTPUT_FILE}")