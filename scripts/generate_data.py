"""Генерация синтетического макроэкономического датасета для прогнозирования PMI."""

import numpy as np
import pandas as pd


def generate_data(n_periods: int = 120) -> pd.DataFrame:
    """Векторизованная генерация датасета с пороговыми эффектами и лагами.

    Args:
        n_periods: Количество месяцев для генерации (минимум 36).

    Returns:
        DataFrame с сгенерированными данными.
    """
    np.random.seed(42)

    # 1. Генерация сырых признаков (uniform distribution)
    data = pd.DataFrame(
        {
            "inflation": np.random.uniform(2.0, 15.0, n_periods),
            "key_rate": np.random.uniform(7.0, 21.0, n_periods),
            "oil_price": np.random.uniform(40.0, 100.0, n_periods),
            "consumer_confidence": np.random.uniform(80.0, 120.0, n_periods),
    }
    )

    # 2. Даты — n месяцев назад от текущей даты
    data["date"] = pd.date_range(end=pd.Timestamp.today(), periods=n_periods, freq="MS"
    )

    # 3. Лаги (векторизовано)
    data["inflation_lag1"] = data["inflation"].shift(1).fillna(data["inflation"])
    data["oil_price_lag1"] = data["oil_price"].shift(1).fillna(data["oil_price"])

    # 4. Пороговый коэффициент key_rate
    coef_rate = np.where(data["inflation"] > 10, -1.0, -0.5)

    # 5. Расчёт PMI по частям (векторизовано)
    base = 55
    term_inf = -0.8 * np.maximum(0, data["inflation"] - 5)
    term_rate = coef_rate * (data["key_rate"] - 10)
    term_oil = 0.04 * (data["oil_price"] - 70)
    term_conf = 0.2 * (data["consumer_confidence"] - 100)
    noise = np.random.normal(0, 1.5, n_periods)

    # Шок нефти: если oil_price < 50, вычитаем 3.0
    term_oil_shock = np.where(data["oil_price"] < 50, -3.0, 0.0)

    pmi_raw = (
        base + term_inf + term_rate + term_oil + term_oil_shock + term_conf + noise
    )
    data["pmi"] = np.clip(pmi_raw, 0, 100)

    # 6. float32 для экономии памяти
    float_cols = [
        "inflation",
        "key_rate",
        "oil_price",
        "consumer_confidence",
        "inflation_lag1",
        "oil_price_lag1",
        "pmi"]
    data[float_cols] = data[float_cols].astype("float32")

    # 7. Порядок колонок
    data = data[
        [
            "date", 
            "inflation", 
            "key_rate", 
            "oil_price",
            "consumer_confidence", 
            "inflation_lag1", 
            "oil_price_lag1", 
            "pmi"
        ]
    ]

    return data


def main() -> None:
    """Точка входа: генерация и сохранение датасета."""
    data = generate_data(n_periods=120)
    output_path = "data/raw/macro.csv"
    data.to_csv(output_path, float_format="%.3f", index=False)
    print(f"Dataset saved to {output_path} ({len(data)} rows)")


if __name__ == "__main__":
    main()
