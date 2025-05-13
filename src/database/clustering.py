from typing import Dict, List, Optional
import math
import random

class KMeansClustering:
    def __init__(self, metrics: List[str], n_clusters: int = 3, max_iter: int = 100):
        self.metrics = metrics
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, teams: List[Dict]):
        """Обучает модель K-means на данных команд"""
        points = [self._extract_metrics(team) for team in teams]
        
        if len(points) < self.n_clusters:
            raise ValueError("Недостаточно данных для заданного числа кластеров")

        # Инициализация случайных центров
        self.centroids = random.sample(points, self.n_clusters)

        for _ in range(self.max_iter):
            clusters = [[] for _ in range(self.n_clusters)]

            # Назначение точек ближайшему центру
            for point in points:
                cluster_idx = self._assign_cluster(point)
                clusters[cluster_idx].append(point)

            # Обновление центров
            new_centroids = []
            for cluster in clusters:
                if not cluster:
                    # Если кластер пустой, оставляем старый центр
                    new_centroids.append(random.choice(points))
                else:
                    new_centroids.append(self._compute_centroid(cluster))

            # Проверка сходимости
            if self._centroids_equal(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def cluster_teams(self, teams: List[Dict]) -> Dict[str, List[Dict]]:
        """Кластеризует команды и возвращает словарь с названиями кластеров"""
        if self.centroids is None:
            raise ValueError("Model not fitted. Call fit() before clustering.")

        # Сопоставляем индексы кластеров с именами 'best', 'middle', 'worst'
        sorted_indices = self._rank_centroids_by_strength()

        # Группируем команды
        clusters = {i: [] for i in range(self.n_clusters)}
        for team in teams:
            metrics = self._extract_metrics(team)
            cluster_idx = self._assign_cluster(metrics)
            clusters[cluster_idx].append(team)

        # Переименовываем кластеры
        labeled_clusters = {
            'best': clusters[sorted_indices[0]],
            'middle': clusters[sorted_indices[1]],
            'worst': clusters[sorted_indices[2]]
        }

        return labeled_clusters

    def _extract_metrics(self, team: Dict) -> Dict[str, float]:
        return {
            metric: self._safe_float(team.get(metric))
            for metric in self.metrics
        }

    @staticmethod
    def _safe_float(value) -> float:
        try:
            return float(value) if value is not None else 1.0
        except (TypeError, ValueError):
            return 1.0

    def _assign_cluster(self, point: Dict[str, float]) -> int:
        distances = [
            self._euclidean_distance(point, centroid)
            for centroid in self.centroids
        ]
        return distances.index(min(distances))

    def _compute_centroid(self, points: List[Dict[str, float]]) -> Dict[str, float]:
        centroid = {}
        for metric in self.metrics:
            centroid[metric] = sum(p[metric] for p in points) / len(points)
        return centroid

    def _euclidean_distance(self, point_a: Dict[str, float], point_b: Dict[str, float]) -> float:
        return math.sqrt(sum(
            (point_a[key] - point_b[key]) ** 2
            for key in self.metrics
            if key in point_a and key in point_b
        ))

    def _centroids_equal(self, c1, c2):
        """Проверяет, совпадают ли центры"""
        if c1 is None or c2 is None:
            return False
        for p1, p2 in zip(c1, c2):
            if any(abs(p1[k] - p2[k]) > 1e-6 for k in self.metrics):
                return False
        return True

    def _rank_centroids_by_strength(self) -> List[int]:
        """
        Возвращает индексы кластеров от лучшего к худшему
        Основа — сумма нормализованных метрик (например, xG, голы и т.п.)
        """
        strengths = []
        for idx, centroid in enumerate(self.centroids):
            strength = sum(centroid.values())
            strengths.append((idx, strength))

        # Сортировка по убыванию силы
        strengths.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in strengths]
    
if __name__ == "__main__":
    # Пример данных (команды с их статистикой)
    data = [
        {"name": "Team A", "attack": 80, "defense": 70},
        {"name": "Team B", "attack": 75, "defense": 75},
        {"name": "Team C", "attack": 20, "defense": 30},
        {"name": "Team D", "attack": 25, "defense": 25},
        {"name": "Team E", "attack": 50, "defense": 45},
        {"name": "Team F", "attack": 45, "defense": 50},
    ]

    # Создаем модель
    model = KMeansClustering(metrics=["attack", "defense"])

    # Обучаем модель
    model.fit(data)

    # Делаем кластеризацию
    clustered_teams = model.cluster_teams(data)

    # Выводим результат
    print("Лучшие команды:", [team["name"] for team in clustered_teams["best"]])
    print("Средние команды:", [team["name"] for team in clustered_teams["middle"]])
    print("Худшие команды:", [team["name"] for team in clustered_teams["worst"]])