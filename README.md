# Emotion recognition (FER-2013)

Модель распознавания эмоций по FER-2013. Код содержит тренировочный Colab notebook и `main.py` — скрипт для инференса с веб-камеры.

## Файлы
- `notebook.ipynb` — Colab ноутбук (preprocessing, training, evaluation).
- `main.py` — realtime inference (OpenCV + загруженная модель `best_model.keras`).
- `requirements.txt`, `.gitignore`.

## Как запустить (локально)
1. Установить зависимости: `pip install -r requirements.txt`
2. Положить `best_model.keras` рядом с `main.py` или обновить путь в скрипте.
3. Запуск: `python main.py`
