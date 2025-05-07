class Clustering:
    def __init__(self, metrics):
        self.metrics = metrics

        self.reference_points = {
            'best': None,
            'middle': None,
            'worst': None
        }

    def set_reference_points_by_teams(self, best_team, middle_team, worst_team):
        self.reference_points['best'] = self._extract_metrics(best_team)
        self.reference_points['middle'] = self._extract_metrics(middle_team)
        self.reference_points['worst'] = self._extract_metrics(worst_team)

    def cluster_teams(self, teams: list) -> dict:
        clusters = {
            'best': [],
            'middle': [],
            'worst': []
        }

        for team in teams:
            metrics = self._extract_metrics(team)
            cluster = self._assign_cluster(metrics)
            clusters[cluster].append(team)

        return clusters

    def _extract_metrics(self, team: dict) -> dict:
        """Извлекает только указанные метрики из данных команды, заменяя None на 1"""
        metrics = {}
        for metric in self.metrics:
            value = team.get(metric)
            try:
                metrics[metric] = float(value) if value is not None else 1.0
            except (TypeError, ValueError):
                metrics[metric] = 1.0
        return metrics

    def _assign_cluster(self, metrics: dict) -> str:
        distances = {}

        for cluster_name, ref_point in self.reference_points.items():
            if ref_point is None:
                continue
            distance = self._euclidean_distance(metrics, ref_point)
            distances[cluster_name] = distance

        return min(distances, key=distances.get)

    def _euclidean_distance(self, point_a: dict, point_b: dict) -> float:
        return sum((point_a[key] - point_b[key]) ** 2 for key in self.metrics) ** 0.5