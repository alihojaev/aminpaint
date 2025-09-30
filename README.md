Anime Manga Inpainting (LaMa) — RunPod Serverless

Проект для деплоя модели `dreMaz/AnimeMangaInpainting` (LaMa) на RunPod Serverless. Сервер реализован на FastAPI и предоставляет endpoint `/inpaint`.

Содержимое:
- `Dockerfile` — сборка контейнера на базе `python:3.10-slim`, установка зависимостей и предзагрузка модели
- `requirements.txt` — список зависимостей
- `server.py` — FastAPI сервер с endpoint `/inpaint`
- `runpod.yaml` — конфигурация для RunPod Serverless

Сборка и локальный запуск

1. Сборка Docker-образа:
   ```bash
   docker build -t anime_manga_inpaint .
   ```

2. Запуск контейнера:
   ```bash
   docker run -p 8000:8000 anime_manga_inpaint
   ```

После запуска сервер будет доступен на `http://localhost:8000`.

API

- `POST /inpaint`
  Вход (JSON):
  ```json
  {
    "image": "<base64>",
    "mask": "<base64>"
  }
  ```
  Выход (JSON):
  ```json
  {
    "result": "<base64>"
  }
  ```

Пример запроса через curl

```bash
curl -X POST http://localhost:8000/inpaint \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64>","mask":"<base64>"}'
```

Деплой на RunPod Serverless

1. Запушьте репозиторий в GitHub (или другой Git-провайдер).
2. В RunPod создайте Serverless Endpoint, укажите Docker-репозиторий или GitHub Actions для сборки.
3. Файл `runpod.yaml` содержит:

```yaml
handler: server.py
entrypoint: uvicorn server:app --host 0.0.0.0 --port 8000
```

Пример запроса на прод-эндпоинт:

```bash
curl -X POST https://YOUR_RUNPOD_ENDPOINT/inpaint \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64>","mask":"<base64>"}'
```

Примечания

- Модель `dreMaz/AnimeMangaInpainting` предзагружается при сборке образа в `/models/AnimeMangaInpainting`, что ускоряет холодный старт.
- Если mask и image разных размеров, mask автоматически масштабируется под размер исходного изображения.
- При наличии GPU сервер автоматически использует CUDA; в противном случае работает на CPU.


