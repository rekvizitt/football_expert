from typing import Dict, List, Optional
import math

class Clustering:
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.reference_points: Dict[str, Optional[Dict[str, float]]] = {
            'best': None,
            'middle': None,
            'worst': None
        }

    def set_reference_points_by_teams(self, best_team: Dict, middle_team: Dict, worst_team: Dict) -> None:
        self.reference_points['best'] = self._extract_metrics(best_team)
        self.reference_points['middle'] = self._extract_metrics(middle_team)
        self.reference_points['worst'] = self._extract_metrics(worst_team)

    def cluster_teams(self, teams: List[Dict]) -> Dict[str, List[Dict]]:
        if any(v is None for v in self.reference_points.values()):
            raise ValueError("Reference points not set. Call set_reference_points_by_teams() first.")
        
        clusters = {k: [] for k in self.reference_points.keys()}
        
        for team in teams:
            metrics = self._extract_metrics(team)
            cluster = self._assign_cluster(metrics)
            clusters[cluster].append(team)
        
        return clusters
    
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

    def _assign_cluster(self, metrics: Dict[str, float]) -> str:
        distances = {
            name: self._euclidean_distance(metrics, ref)
            for name, ref in self.reference_points.items()
            if ref is not None
        }
        return min(distances, key=distances.get)

    def _euclidean_distance(self, point_a: Dict[str, float], point_b: Dict[str, float]) -> float:
        return math.sqrt(sum(
            (point_a[key] - point_b[key]) ** 2
            for key in self.metrics
            if key in point_a and key in point_b
        ))