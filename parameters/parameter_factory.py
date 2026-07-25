from parameters.parameter_transformer import ParameterTransformer
from parameters.parameter_specifications import (
    GAUSSIAN_LOCAL_EXPLOSIONS_SPEC,
    STUDENT_LOCAL_EXPLOSIONS_SPEC,
    EGB2_LOCAL_EXPLOSIONS_SPEC,
)


class ParameterTransformerFactory:

    _registry = {
        "n": GAUSSIAN_LOCAL_EXPLOSIONS_SPEC,
        "t": STUDENT_LOCAL_EXPLOSIONS_SPEC,
        "egb2": EGB2_LOCAL_EXPLOSIONS_SPEC,
    }


    @classmethod
    def create(cls, name: str):

        key = name.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown parameter specification '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )

        return ParameterTransformer(cls._registry[key])