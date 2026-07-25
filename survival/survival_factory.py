from typing import Callable
from survival.survival_conditions import g1, g2, g3

class SurvivalFactory:

    _functions = {
        "g1": g1,
        "g2": g2,
        "g3": g3,
    }

    @staticmethod
    def create(name) -> Callable:
        try:
            return SurvivalFactory._functions[name.lower()]
        except KeyError:
            raise ValueError(f"Unknown survival function '{name}'")