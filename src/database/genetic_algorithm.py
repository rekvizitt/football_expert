import random
from typing import List, Dict, Any, Callable
from functools import lru_cache
from src.logger import logger

class GeneticDataSorter:
    def __init__(self):
        self.population_size = 50
        self.max_generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.tournament_size = 3
        self.elitism_count = 2

    def _generate_individual(self, columns: List[str], weights: List[float]) -> Dict:
        return {col: random.uniform(-1, 1) * w for col, w in zip(columns, weights)}

    @lru_cache(maxsize=None)
    def _evaluate_individual(self, individual_frozen: frozenset, data_frozen: frozenset, target_func: Callable[[Dict], float]) -> float:
        individual = dict(individual_frozen)
        data = [dict(item) for item in data_frozen]
        if not data:
            return -float('inf')
        sorted_data = self._sort_with_weights(data, individual)
        sample_size = min(100, len(sorted_data))
        sample = sorted_data[:sample_size]
        
        return sum(target_func(item) for item in sample) / sample_size

    def _sort_with_weights(self, data: List[Dict], weights: Dict[str, float]) -> List[Dict]:
        def sort_key(item: Dict) -> float:
            score = 0.0
            for col, weight in weights.items():
                if col in item and item[col] is not None:
                    try:
                        score += float(item[col]) * weight
                    except (ValueError, TypeError):
                        score += weight * (1 if item[col] else -1)
            return score
        return sorted(data, key=sort_key, reverse=True)
    
    def _tournament_selection(self, population: List[Dict], fitness: List[float]) -> Dict:
        selected = random.sample(list(zip(population, fitness)), self.tournament_size)
        selected.sort(key=lambda x: x[1], reverse=True)
        return selected[0][0]
    
    def _mutate_individual(self, individual: Dict, mutation_rate: float) -> Dict:
        mutated = individual.copy()
        for col in mutated:
            if random.random() < mutation_rate:
                mutated[col] += random.uniform(-0.5, 0.5)
                mutated[col] = max(-1, min(1, mutated[col]))
        return mutated
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        child = {}
        for col in parent1:
            if random.random() < 0.5:
                child[col] = (parent1[col] + parent2[col]) / 2
            else:
                child[col] = parent1[col] if random.random() < 0.5 else parent2[col]
        return child
    
    def optimize_sorting(self, data: List[Dict],
                         target_func: Callable[[Dict], float],
                         columns: List[str] = None,
                         weights: List[float] = None,
                         generations: int = None) -> List[Dict]:
        if not data:
            return data
        
        if columns is None:
            columns = [k for k, v in data[0].items()
                       if isinstance(v, (int, float)) and v is not None]
            
        if not columns:
            return data
        
        if weights is None:
            weights = [1.0] * len(columns)
        
        if generations is None:
            generations = self.max_generations
        
        population = [self._generate_individual(columns, weights) for _ in range(self.population_size)]
        
        best_individual = None
        best_fitness = -float('inf')
        
        for gen in range(generations):
            fitness = [self._evaluate_individual(frozenset(ind.items()), frozenset(tuple(item.items()) for item in data), target_func) for ind in population]
            
            current_best = max(fitness)
            if current_best > best_fitness: 
                best_fitness = current_best
                best_index = fitness.index(current_best)
                best_individual = population[best_index]
                logger.debug(f'Generation {gen}: Improved fitness to {best_fitness:.4f}')
            new_population = []
            elite_indicies = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)[:self.elitism_count]
            new_population.extend([population[i] for i in elite_indicies])
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                
                child = self._crossover(parent1, parent2)
                child = self._mutate_individual(child, self.mutation_rate)
                new_population.append(child)
            population = new_population
        return self._sort_with_weights(data, best_individual)
    
    def filter_and_sort(self, data: List[Dict],
                        target_func: Callable[[Dict], float],
                        filter_params: Dict[str, Any] = None,
                        limit: int = None,
                        columns: List[str] = None,
                        weights: List[float] = None) -> List[Dict]:
        if not data:
            return []
        filtered_data = data[:]
        if filter_params:
            for col, val in filter_params.items():
                filtered_data = [
                    item for item in filtered_data
                    if str(val).lower() in str(item.get(col, '')).lower()
                ]
        if limit:
            filtered_data = filtered_data[:limit]
            
        if not filtered_data:
            return []
        
        return self.optimize_sorting(filtered_data, target_func, columns, weights)

if __name__ == "__main__":
    # Example usage
    def example_target_func(item: Dict):
        xg = item.get('xg_last5', 0) or item.get('xg_total', 0) or 0
        performance = item.get('performance_ga', 0) or 0
        return 0.7 * xg + 0.3 * (performance / 100)

    data = [
        {"match_id": 1, "xg_last5": 2.1, "performance_ga": 80},
        {"match_id": 2, "xg_last5": 1.5, "performance_ga": 90},
        {"match_id": 3, "xg_last5": 3.0, "performance_ga": 70},
    ]

    Gds = GeneticDataSorter()
    sorted_data = Gds.filter_and_sort(
        data=data,
        target_func=example_target_func,
        columns=["xg_last5", "performance_ga"],
        weights=[0.7, 0.3]
    )
