import subprocess
import urllib.request
import json
import time
import sys

def check_ollama():
    """Проверяет, запущен ли Ollama и какие модели доступны."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
            return True, [m["name"] for m in data.get("models", [])]
    except Exception:
        return False, []

def get_vram():
    """Считывает использование VRAM через nvidia-smi."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,name",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 3:
            used = float(parts[0])
            total = float(parts[1])
            name = parts[2]
            return name, used, total
    except FileNotFoundError:
        print("⚠️ Команда nvidia-smi не найдена. Установите драйверы NVIDIA.")
    except Exception as e:
        print(f"⚠️ Ошибка чтения GPU: {e}")
    return None, None, None

def main():
    print("🔍 Проверка статуса Ollama и загрузки VRAM...")
    time.sleep(2)  # Даем серверу время на инициализацию

    is_running, models = check_ollama()
    if is_running:
        print(f"✅ Ollama работает. Доступные модели: {models if models else 'нет загруженных'}")
    else:
        print("❌ Ollama не отвечает. Запустите start_ollama.bat и попробуйте снова.")
        return

    name, used, total = get_vram()
    if name:
        print(f"🖥️  GPU: {name}")
        print(f"💾 VRAM: {used:.0f} MB / {total:.0f} MB (свободно: {total-used:.0f} MB)")
        
        if used > 1500:
            print("✅ Модель загружена в VRAM и готова к работе.")
        else:
            print("⚠️ VRAM почти свободна. Модель не загружена или работает на CPU.")
            print("   → Для загрузки выполните: curl http://localhost:11434/api/generate -d '{\"model\":\"qwen2.5:3b\",\"prompt\":\"test\"}'")
    else:
        print("⚠️ Не удалось считать VRAM. Проверьте драйверы NVIDIA.")

if __name__ == "__main__":
    main()