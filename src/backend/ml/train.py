import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
import os

# =============================================================================
# 1. Кастомный трансформер для создания взаимодействия (Interaction)
# =============================================================================
class FeatureInteraction(BaseEstimator, TransformerMixin):
    def __init__(self, col1: str, col2: str, feature_columns: list = None):
        self.col1 = col1
        self.col2 = col2
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.col1_idx_ = None
        self.col2_idx_ = None

    def fit(self, X, y=None):
        # Индексы фиксируются по порядку feature_columns
        self.col1_idx_ = self.feature_columns.index(self.col1)
        self.col2_idx_ = self.feature_columns.index(self.col2)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            interaction = X[self.col1] * X[self.col2]
            new_col_name = f"{self.col1}_x_{self.col2}"
            result = X.copy()
            result[new_col_name] = interaction
            return result
        else:
            # Работаем с numpy array (после RobustScaler)
            col1_vals = X[:, self.col1_idx_]
            col2_vals = X[:, self.col2_idx_]
            interaction = col1_vals * col2_vals
            new_col = interaction.reshape(-1, 1)
            return np.hstack([X, new_col])


# =============================================================================
# 2. Загрузка данных
# =============================================================================
def load_data(data_path: str = "data/raw/macro.csv") -> pd.DataFrame:
    """Загрузка макроэкономических данных из CSV."""
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


# =============================================================================
# 3. Разделение на признаки и целевую переменную
# =============================================================================
FEATURE_COLUMNS = ['inflation', 'key_rate', 'oil_price', 'consumer_confidence',
                   'inflation_lag1', 'oil_price_lag1']
TARGET_COLUMN = 'pmi'


def prepare_features(df: pd.DataFrame):
    """Подготовка матрицы признаков и целевой переменной."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


# =============================================================================
# 4. Сборка и обучение пайплайна
# =============================================================================
def build_pipeline():
    """Сборка sklearn Pipeline с RobustScaler, FeatureInteraction и CatBoost."""
    pmis_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('interaction', FeatureInteraction('inflation', 'key_rate', FEATURE_COLUMNS)),
        ('regressor', CatBoostRegressor(
            verbose=0,
            random_state=42,
            allow_writing_files=False
        ))
    ])
    return pmis_pipeline


def train_model(pipeline, X_train, y_train, X_test, y_test):
    """Обучение пайплайна и оценка метрик."""
    pipeline.fit(X_train, y_train)

    # Предсказания
    y_pred = pipeline.predict(X_test)

    # Метрики
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    print("\n" + "=" * 50)
    print("МЕТРИКИ МОДЕЛИ НА TEST SET")
    print("=" * 50)
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.4f}%")
    print("=" * 50 + "\n")

    return metrics


# =============================================================================
# 5. Сохранение модели
# =============================================================================
def save_model(pipeline, output_path: str = "model.pkl"):
    """Сохранение всего объекта Pipeline целиком через joblib."""
    joblib.dump(pipeline, output_path)
    print(f"Model pipeline successfully saved to {output_path}")


# =============================================================================
# 6. Интерпретация — SHAP values (beeswarm plot)
# =============================================================================
def generate_shap_plot(pipeline, X_train, output_dir: str = "reports"):
    """Построение и сохранение SHAP beeswarm plot."""
    os.makedirs(output_dir, exist_ok=True)

    regressor = pipeline.named_steps['regressor']
    # Пропускаем X_train через трансформеры пайплайна (scaler + interaction)
    X_transformed = pipeline[:-1].transform(X_train)

    explainer = shap.Explainer(regressor)
    shap_values = explainer(X_transformed)

    plot_path = os.path.join(output_dir, "shap_plot.png")
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"SHAP plot successfully saved to {plot_path}")


# =============================================================================
# 7. Main
# =============================================================================
def main():
    # Загрузка данных
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw', 'macro.csv')
    if not os.path.exists(data_path):
        data_path = "data/raw/macro.csv"

    print(f"Loading data from: {data_path}")
    df = load_data(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

    # Разделение на признаки и target
    X, y = prepare_features(df)

    # Train/test split (time-aware: first 80% train, last 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")

    # Сборка пайплайна
    pmis_pipeline = build_pipeline()

    # Обучение и оценка
    metrics = train_model(pmis_pipeline, X_train, y_train, X_test, y_test)

    # Сохранение модели
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'model.pkl')
    save_model(pmis_pipeline, model_path)

    # SHAP визуализация
    generate_shap_plot(pmis_pipeline, X_train)

    print("\nDone! Model and SHAP plot saved.")


if __name__ == "__main__":
    main()
