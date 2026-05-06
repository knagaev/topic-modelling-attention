
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch
#import pandas as pd

def filter_newsgroups_strict(newsgroups, leave_config, random_state=42):
    """
    Оставляет ТОЛЬКО указанные категории и строго заданное количество статей в них.
    Все остальные категории удаляются полностью.
    
    Args:
        newsgroups: объект Bunch из fetch_20newsgroups()
        leave_config: dict {category_name: n_to_leave}
                      Пример: {'rec.motorcycles': 100, 'sci.med': 50}
        random_state: seed для воспроизводимости
        
    Returns:
        Bunch с отфильтрованными данными
    """
    rng = np.random.default_rng(random_state)
    
    # Инициализируем маску "удалить всё"
    mask_keep = np.zeros(len(newsgroups.data), dtype=bool)
    
    print("⚙️ Фильтрация датасета...")
    
    for category, n_leave in leave_config.items():
        if category not in newsgroups.target_names:
            raise ValueError(f"❌ Категория '{category}' не найдена.\nДоступные: {newsgroups.target_names}")
            
        cat_idx = newsgroups.target_names.index(category)
        cat_indices = np.where(newsgroups.target == cat_idx)[0]
        n_available = len(cat_indices)
        
        if n_leave > n_available:
            raise ValueError(f"❌ В категории '{category}' всего {n_available} статей, нельзя оставить {n_leave}.")
        
        if n_leave < 0:
            raise ValueError(f"❌ n_to_leave не может быть отрицательным.")

        # Выбираем случайные индексы для сохранения
        if n_leave > 0:
            indices_to_keep = rng.choice(cat_indices, size=n_leave, replace=False)
            mask_keep[indices_to_keep] = True
            
        print(f"   • '{category}': оставлено {n_leave} из {n_available}")

    # Формируем итоговый объект
    filtered_data = [newsgroups.data[i] for i in range(len(newsgroups.data)) if mask_keep[i]]
    filtered_target = newsgroups.target[mask_keep]
    filtered_filenames = [newsgroups.filenames[i] for i in range(len(newsgroups.filenames)) if mask_keep[i]]
    
    # Создаем словарь для маппинга индексов на названия категорий (только для тех, что остались)
    # Но target_names оставляем полным списком, чтобы индексы классов совпадали с оригиналом
    # Это важно для совместимости со sklearn классификаторами
    
    result_bunch = Bunch(
        data=filtered_data,
        target=filtered_target,
        filenames=filtered_filenames,
        target_names=newsgroups.target_names,
        DESCR=newsgroups.DESCR
    )
    
    # --- Вывод статистики ---
    print("\n📊 Итоговое распределение статей:")
    unique_targets, counts = np.unique(result_bunch.target, return_counts=True)
    
    stats = []
    for t, c in zip(unique_targets, counts):
        cat_name = result_bunch.target_names[t]
        stats.append({'Category': cat_name, 'Count': c})

    return result_bunch

# 1. Загружаем полный датасет
ng_all = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

# 2. Настраиваем конфиг: какие категории и сколько статей оставить
# Все остальные 18 категорий будут удалены полностью
config = {
    'rec.motorcycles': 3,
    'sci.med': 4,
    'comp.graphics': 1,
    'talk.politics.misc': 2
}

# 3. Применяем фильтр
ng_filtered = filter_newsgroups_strict(ng_all, leave_config=config, random_state=42)
print('ok')