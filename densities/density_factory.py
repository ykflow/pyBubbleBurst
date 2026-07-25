from typing import Callable
from densities.log_densities import gaussian_loglik, student_loglik, egb2_loglik


class DensityFactory:

    _functions = {
        "n": gaussian_loglik,
        "t": student_loglik,
        "egb2": egb2_loglik,
    }

    @staticmethod
    def create(name) -> Callable:
        try:
            return DensityFactory._functions[name.lower()]
        except KeyError:
            raise ValueError(f"Unknown density '{name}'")