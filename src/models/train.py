import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pathlib import Path
import joblib
import json
from src.config import ConfigManager
from src.logger import logger

class ModelTrainer:
    def __init__(self):
        config_manager = ConfigManager()

        self.train_data_dir = config_manager.train_data_dir
        self.models_dir = config_manager.models_dir
        self.metrics_dir = Path(config_manager.metrics_dir)
        
        self.X_train_path = Path(self.train_data_dir) / "X_train.json"
        self.X_test_path = Path(self.train_data_dir) / "X_test.json"
        self.y_train_path = Path(self.train_data_dir) / "y_train.json"
        self.y_test_path = Path(self.train_data_dir) / "y_test.json"
    
    def load_data(self):
        """
        Загрузка данных из JSON-файлов.

        :return: X_train, X_test, y_train, y_test.
        """
        try:
            X_train = pd.read_json(self.X_train_path, orient='records')
            X_test = pd.read_json(self.X_test_path, orient='records')
            y_train = pd.read_json(self.y_train_path, orient='records').values.ravel()
            y_test = pd.read_json(self.y_test_path, orient='records').values.ravel()

            # Remap class labels to be consecutive integers starting from 0
            unique_classes = np.unique(np.concatenate((y_train, y_test)))
            class_mapping = {class_label: i for i, class_label in enumerate(unique_classes)}
            y_train = np.array([class_mapping[label] for label in y_train])
            y_test = np.array([class_mapping[label] for label in y_test])

            logger.info('Данные загружены и классы переопределены')
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f'Ошибка загрузки данных: {e}')
            return None, None, None, None
        
    def train_model(self, model_class, X_train, y_train, **kwargs):
        """
        Обучение модели.

        :param model_class: Класс модели.
        :param X_train: Обучающая выборка признаков.
        :param y_train: Обучающая выборка целевой переменной.
        :param kwargs: Дополнительные параметры для модели.
        :return: Обученная модель.
        """
        try:
            model = model_class(**kwargs)
            model.fit(X_train, y_train)
            logger.debug(f'{model_class.__name__} trained')
            return model
        except Exception as e:
            logger.error(f'Error train {model_class.__name__}: {e}')
            return None
    
    def tune_hyperparameters(self, model_class, X_train, y_train, param_grid):
        """
        Настройка гиперпараметров модели.

        :param model_class: Класс модели.
        :param X_train: Обучающая выборка признаков.
        :param y_train: Обучающая выборка целевой переменной.
        :param param_grid: Сетка гиперпараметров.
        :return: Лучшая модель после настройки.
        """
        try:
            grid_search = GridSearchCV(model_class(), param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            logger.debug(f'Гиперпараметры настроены для {model_class.__name__}. Лучшие параметры: {grid_search.best_params_}')
            return grid_search.best_estimator_
        except Exception as e:
            logger.error(f'Ошибка настройки гиперпараметров: {e}')
            return None

    def evaluate_model(self, model, X_test, y_test):
        """
        Оценка модели.

        :param model: Модель для оценки.
        :param X_test: Тестовая выборка признаков.
        :param y_test: Тестовая выборка целевой переменной.
        :return: Точность, F1-мера, ROC-AUC.
        """
        try:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
            logger.info(f'Точность: {accuracy:.3f}, F1-мера: {f1:.3f}, ROC-AUC: {roc_auc:.3f}')
            return accuracy, f1, roc_auc
        except Exception as e:
            logger.error(f'Ошибка оценки модели: {e}')
            return None, None, None
    
    def save_model(self, model, model_name):
        """
        Сохранение модели.

        :param model: Модель для сохранения.
        :param model_name: Имя модели.
        """
        try:
            model_path = self.models_dir / f'{model_name}.pkl'
            joblib.dump(model, model_path)
            logger.info(f'Модель {model_name} сохранена в {model_path}')
        except Exception as e:
            logger.error(f'Ошибка сохранения модели: {e}')
    
    def save_metrics(self, metrics, model_name):
        """
        Сохранение метрик.

        :param metrics: Словарь с метриками.
        :param model_name: Имя модели.
        """
        try:
            metrics_path = self.metrics_dir / f'{model_name}_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            logger.info(f'Метрики модели {model_name} сохранены в {metrics_path}')
        except Exception as e:
            logger.error(f'Ошибка сохранения метрик: {e}')

    def train_and_save_models(self):
        """
        Обучение, настройка и сохранение моделей.
        """
        # Загрузка данных
        X_train, X_test, y_train, y_test = self.load_data()
        if X_train is None or X_test is None or y_train is None or y_test is None:
            logger.error('Ошибка загрузки данных. Завершение работы.')
            return

        # Определение моделей и их параметров
        models = {
            'logistic_regression': (LogisticRegression, {'max_iter': 5000, 'solver': 'saga'}),
            'random_forest': (RandomForestClassifier, {'n_estimators': 100, 'random_state': 42}),
            'gradient_boosting': (GradientBoostingClassifier, {'n_estimators': 100, 'random_state': 42}),
            'xgboost': (XGBClassifier, {'n_estimators': 100, 'random_state': 42, 'eval_metric': 'logloss'}),
        }

        # Настройка гиперпараметров
        param_grids = {
            'logistic_regression': {'C': [0.1, 1, 10]},
            'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
            'gradient_boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
            'xgboost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
        }

        # Обучение, настройка и оценка моделей
        for model_name, (model_class, kwargs) in models.items():
            model = self.train_model(model_class, X_train, y_train, **kwargs)
            if model is not None:
                tuned_model = self.tune_hyperparameters(model_class, X_train, y_train, param_grids[model_name])
                if tuned_model is not None:
                    accuracy, f1, roc_auc = self.evaluate_model(tuned_model, X_test, y_test)
                    metrics = {'accuracy': accuracy, 'f1': f1, 'roc_auc': roc_auc}
                    self.save_metrics(metrics, model_name)
                    self.save_model(tuned_model, model_name)
                    
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_save_models()