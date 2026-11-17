# 01_data_generation.md

Я создал файл `scripts/generate_data.py` и `scripts/requirements.txt`. Ниже описываю логику его работы.

**Заголовки функций из `generate_data.py`:**
- `generate_data(n_periods: int = 120)` — векторизованная генерация датасета
- `main()` — точка входа, сохранение в CSV

## Step 1: Создание зависимостей

В папке `scripts/` создан файл `requirements.txt` с зависимостями `numpy` и `pandas`.

*Rationale:* Навык `dataset-generation-skill.md` требует, чтобы у скрипта генерации данных был собственный файл зависимостей. Это обеспечивает изоляцию и воспроизводимость окружения. Перед запуском скрипта зависимости должны быть установлены из этого файла.

## Step 2: Выбор подхода к генерации данных

Использован **векторизованный подход** через `numpy` и `pandas` без циклов `for`/`while`.

*Rationale:* Векторизованные операции работают на C-уровне, что обеспечивает высокую производительность. Циклы по строкам категорически запрещены спецификацией навыка.

## Step 3: Генерация сырых признаков

```python
data = pd.DataFrame({
    "inflation": np.random.uniform(2.0, 15.0, n_periods),
    "key_rate": np.random.uniform(7.0, 21.0, n_periods),
    "oil_price": np.random.uniform(40.0, 100.0, n_periods),
    "consumer_confidence": np.random.uniform(80.0, 120.0, n_periods),
})
```

*Rationale:* Uniform-распределение обеспечивает равномерное покрытие пространства признаков без смещения в какую-либо область, что важно для обучения модели на разнообразных сценариях. Диапазоны соответствуют спецификации: inflation [2.0, 15.0], key_rate [7.0, 21.0], oil_price [40.0, 100.0], consumer_confidence [80.0, 120.0].

## Step 4: Создание дат

```python
data["date"] = pd.date_range(
    end=pd.Timestamp.today(), periods=n_periods, freq="MS"
)
```

*Rationale:* `freq="MS"` (month start) создаёт последовательность с 1-го числа каждого месяца. Даты идут от текущей даты назад, что соответствует требованию "минимум 36 месяцев от текущей даты назад". Использовано 120 периодов — верхняя граница рекомендуемого диапазона 60-120.

## Step 5: Расчёт лагов

```python
data["inflation_lag1"] = data["inflation"].shift(1).fillna(data["inflation"])
data["oil_price_lag1"] = data["oil_price"].shift(1).fillna(data["oil_price"])
```

*Rationale:* `shift(1)` сдвигает значения на одну строку вниз, создавая лаг предыдущего месяца. Для первой строки `fillna` подставляет текущее значение, как требует спецификация: "Для первой строки лаги равны значениям текущего месяца".

## Step 6: Пороговые эффекты

```python
coef_rate = np.where(data["inflation"] > 10, -1.0, -0.5)
term_oil_shock = np.where(data["oil_price"] < 50, -3.0, 0.0)
```

*Rationale:* Пороговые эффекты вводят нелинейность — экономика реагирует на изменения ставок по-разному при низкой и высокой инфляции. Если `inflation <= 10`, коэффициент `-0.5`; если `inflation > 10`, коэффициент `-1.0` (эффект удваивается). Шок нефти моделирует кризисный сценарий: при `oil_price < 50` из PMI вычитается дополнительно `3.0` пункта.

## Step 7: Расчёт PMI по формуле

```python
base = 55
term_inf = -0.8 * np.maximum(0, data["inflation"] - 5)
term_rate = coef_rate * (data["key_rate"] - 10)
term_oil = 0.04 * (data["oil_price"] - 70)
term_conf = 0.2 * (data["consumer_confidence"] - 100)
noise = np.random.normal(0, 1.5, n_periods)

pmi_raw = base + term_inf + term_rate + term_oil + term_oil_shock + term_conf + noise
data["pmi"] = np.clip(pmi_raw, 0, 100)
```

*Rationale:* Линейная комбинация с нелинейными порогами и шумом имитирует реальное поведение PMI. 
- `term_inf`: инфляция выше 5% снижает PMI с коэффициентом -0.8.
- `term_rate`: ставка выше 10% снижает PMI, сила влияния зависит от порога инфляции.
- `term_oil`: цена нефти выше 70 повышает PMI, ниже — снижает.
- `term_conf`: уверенность потребителей выше 100 повышает PMI.
- `noise`: макроэкономические шоки обычно распределены нормально, поэтому используется `np.random.normal(0, 1.5)`.
- `np.clip` гарантирует, что значения остаются в допустимом диапазоне [0, 100].

## Step 8: Сохранение в CSV

```python
float_cols = ["inflation", "key_rate", "oil_price", "consumer_confidence",
              "inflation_lag1", "oil_price_lag1", "pmi"]
data[float_cols] = data[float_cols].astype("float32")

data.to_csv("data/raw/macro.csv", float_format="%.3f", index=False)
```

*Rationale:* `float32` экономит память (~50% по сравнению с float64). `float_format="%.3f"` обеспечивает точность до 3 знаков после запятой. Порядок колонок строго соответствует спецификации: `date`, `inflation`, `key_rate`, `oil_price`, `consumer_confidence`, `inflation_lag1`, `oil_price_lag1`, `pmi`.

## Итог

- **Скрипт:** `scripts/generate_data.py` — генерация датасета
- **Зависимости:** `scripts/requirements.txt` — numpy, pandas
- **Результат:** `data/raw/macro.csv` — 120 строк, 8 колонок, float32
- **Запуск:** `python scripts/generate_data.py`
