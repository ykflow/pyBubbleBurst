from typing import Callable
from scores.score_functions import gaussian_score, student_score, egb2_score


class ScoreFactory:

    _functions = {
        "n": gaussian_score,
        "t": student_score,
        "egb2": egb2_score,
    }

    @staticmethod
    def create(name) -> Callable:
        try:
            return ScoreFactory._functions[name.lower()]
        except KeyError:
            raise ValueError(f"Unknown survival function '{name}'")