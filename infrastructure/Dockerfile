FROM python:3.11-slim

# Устанавливаем пользователя appuser
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -s /sbin/nologin appuser

WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY src/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY src/backend/ .

# Модель монтируется через volume при запуске (docker-compose)

# Переключаемся на non-root пользователя
USER appuser

EXPOSE 8000

# Запуск через uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port 8000
