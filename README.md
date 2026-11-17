# PMI Forecasting Service

MLOps-сервис для прогнозирования индекса PMI с учетом пороговых эффектов макроэкономики.

## Архитектура

Проект является монорепозиторием, разделенным на независимые слои:

- **Data** — генерация и хранение сырых/обработанных данных
- **ML** — обучение моделей (CatBoost)
- **Backend API** — FastAPI-сервер для предсказаний
- **Frontend** — интерфейс с интерактивными слайдерами (Vanilla JS + Chart.js)

## Развертывание

Проект запускается одной командой:

```bash
docker-compose up --build
```

## Документация

Подробное описание каждого этапа разработки:

| Файл | Описание |
|------|----------|
| [01_data_generation.md](docs/01_data_generation.md) | Лог генерации данных, формула PMI, пороговые эффекты |
| [02_model_training.md](docs/02_model_training.md) | Метрики (RMSE/MAE), график SHAP, обоснование выбора CatBoost |
| [03_api_design.md](docs/03_api_design.md) | Схема эндпоинтов, примеры JSON Request/Response |
| [04_frontend_logic.md](docs/04_frontend_logic.md) | Описание работы JS-слайдеров, связи с API, Chart.js |
| [05_deployment.md](docs/05_deployment.md) | Инструкция по docker-compose, переменные .env |
