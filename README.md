# QA_VLM

Web-интерфейс (демо) для мультимодальной модели **SmolVLM2-2.2B-Instruct**, поддерживающий обработку изображений и текстовых запросов.

Проект реализован в виде Docker-приложения и предназначен для демонстрации возможностей мультимодальных LLM.

---

## Возможности

Приложение реализует следующие use-case сценарии:

* **Image Captioning** — генерация текстового описания изображения
* **Visual Question Answering (VQA)** — ответы на вопросы по загруженному изображению без повторной загрузки
* **Optical Character Recognition (OCR)** — извлечение текста с изображения с возможностью скачать результат в `.txt`

> Согласно требованиям ТЗ, captioning / image description / VQA считаются одним сценарием, OCR — отдельным.

---

## Интерфейс

* Web UI реализован с помощью **Gradio**
* Используются вкладки (Tabs) для разных сценариев
* Одно изображение используется повторно для всех сценариев
* Реализована валидация:

  * загрузка только изображений
  * запрет пустых вопросов
  * понятные сообщения об ошибках

---

## Архитектура

* **Модель:** `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
* **Backend:** Python + HuggingFace Transformers
* **Frontend:** Gradio
* **Контейнеризация:** Docker + Docker Compose v2
* **Ускорение:** GPU (CUDA) с fallback на CPU

---

## Требования

### Аппаратные

* GPU с 8 ГБ VRAM (рекомендуется)
  или
* CPU (поддерживается, но медленнее)

### Программные

* Docker ≥ 20.x
* Docker Compose v2
* NVIDIA Container Toolkit (для GPU режима)

---

## Конфигурация через переменные окружения

Все переменные окружения **задаются в файле `.env`**, который используется Docker Compose при запуске приложения.

Пример файла `.env`:

```env
MODEL_NAME=HuggingFaceTB/SmolVLM2-2.2B-Instruct
MODEL_PATH=/models/model
ALLOW_MODEL_DOWNLOAD=yes
MODEL_DEVICE=gpu
HF_TOKEN=hf_xxx
GRADIO_PORT=7860
```

Используемые переменные:

| Переменная             | Описание                    | Значение по умолчанию                  |
| ---------------------- | --------------------------- | -------------------------------------- |
| `MODEL_NAME`           | HuggingFace модель          | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` |
| `MODEL_PATH`           | Путь к весам модели         | `/models/model`                        |
| `ALLOW_MODEL_DOWNLOAD` | Разрешить скачивание модели | `no`                                   |
| `MODEL_DEVICE`         | Режим работы                | `gpu`                                  |
| `HF_TOKEN`             | HuggingFace token           | —                                      |
| `GRADIO_PORT`          | Порт web-интерфейса         | `7860`                                 |

---

## Хранение данных

Проект использует **bind-mount** для прозрачного хранения данных на хосте.

| Назначение  | Путь в контейнере  | Путь на хосте  |
| ----------- | ------------------ | -------------- |
| Веса модели | `/models/model`    | `./model_data` |
| HF cache    | `/models/hf_cache` | `./hf_cache`   |

Веса модели **не хранятся внутри Docker-образа**.

---

## Запуск проекта

### 1. Первый запуск (со скачиванием модели)

1. Создать файл `.env` ( если его нет)
2. Указать в нём `HF_TOKEN` и `ALLOW_MODEL_DOWNLOAD=yes`
3. Запустить:

```bash
cd QA_VLM
```

```bash
docker compose up --build
```

При первом запуске:

* модель скачивается в `./model_data`
* кэш HuggingFace сохраняется в `./hf_cache`

---

### 2. Оффлайн-запуск

В файле `.env`:

```env
ALLOW_MODEL_DOWNLOAD=no
```

Затем:

```bash
docker compose up
```

После этого:

* приложение не обращается к интернету
* используются только локальные файлы

---

## CPU-only режим

В файле `.env`:

```env
MODEL_DEVICE=cpu
```

Затем:

```bash
docker compose up
```

GPU при этом не требуется.

---

## Структура проекта

```text
.
├── app.py              # Web-приложение
├── requirements.txt    # Python зависимости
├── Dockerfile
├── docker-compose.yml
├── .env                # Переменные окружения
├── model_data/         # Веса модели (bind-mount)
├── hf_cache/           # HuggingFace cache
└── README.md
```


## Ограничения

* CPU-режим работает медленно и предназначен для демонстрации

---

## Лицензия

Модель и код используются в соответствии с лицензиями:

* HuggingFace Transformers
* SmolVLM2 (HuggingFaceTB)
