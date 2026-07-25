import numpy as np


class ParameterTransformer:
    """
    Base class for parameter transformations.

    Concrete subclasses only need to define:

        self.param_names
        self.blocks
        self.constraints
        self.inits

    and then call:

        self._build_indices()

    """

    def __init__(self, spec):
        self.spec = spec
        self.param_names = [x[0] for x in spec]
        self._k_unknown = len(self.param_names)
        self.idx = {name: i for i, name in enumerate(self.param_names)}

        self.blocks = {}
        self.continuous = []
        self.discontinuous = []

        self.inits = {}
        self.constraints = {}

        for name, transform, kind, block, init in spec:
            self.blocks.setdefault(block, []).append(name)
            self.inits[name] = init
            self.constraints[name] = transform

            if kind == "continuous":
                self.continuous.append(name)
            elif kind == "discontinuous":
                self.discontinuous.append(name)
            else:
                raise ValueError(f"Unknown parameter type {kind}")

        self._idx_cont = np.array([self.idx[x]for x in self.continuous],dtype=np.intp)
        self._idx_discont = np.array([self.idx[x]for x in self.discontinuous],dtype=np.intp)


    def transform(self,untransformed_vector: np.ndarray) -> np.ndarray:
        transformed = np.empty_like(untransformed_vector, dtype=np.float64)
        for i, name in enumerate(self.param_names):
            forward, _ = self.constraints[name]
            transformed[i] = forward(untransformed_vector[i])
        return transformed


    def untransform(self, transformed_vector: np.ndarray) -> np.ndarray:
        untransformed = np.empty_like(transformed_vector, dtype=np.float64)
        for i, name in enumerate(self.param_names):
            _, inverse = self.constraints[name]
            untransformed[i] = inverse( transformed_vector[i])
        return untransformed

    def split_vector(self,full_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Split vector into discontinuous and continuous:
        """
        return full_vector[self._idx_discont], full_vector[self._idx_cont],

    def merge_params(self, discontinuous_vector: np.ndarray,  continuous_vector: np.ndarray) -> np.ndarray:
        full_vector = np.empty(self._k_unknown, dtype=np.float64)
        full_vector[self._idx_discont] = discontinuous_vector
        full_vector[self._idx_cont] = continuous_vector
        return full_vector

    def split_blocks(self, vector: np.ndarray) -> dict:
        """
        Convert flat parameter vector into:
            theta_mu
            theta_b
            theta_g
            theta_d
        """

        parameters = {}
        for block, names in self.blocks.items():
            parameters[block] = np.array([vector[self.idx[name]] for name in names], dtype=np.float64)
        return parameters

    def get_mle_inits(self) -> np.ndarray:
        """
        Returns initial values in unconstrained space.
        """
        initial = np.array([self.inits[name] for name in self.param_names], dtype=np.float64)
        return self.transform(initial)

    def to_kwargs(self,vector: np.ndarray) -> dict:
        return dict(zip(self.param_names, vector))

    @staticmethod
    def from_pos_to_r(val):
        return np.log(val)

    @staticmethod
    def from_r_to_pos(val):
        return np.exp(val)

    @staticmethod
    def from_1_plus_to_r(val):
        return np.log(val - 1.0)

    @staticmethod
    def from_r_to_1_plus(val):
        return np.exp(val) + 1.0

    @staticmethod
    def from_0_1_to_r(val):
        return np.log(val / (1.0 - val))

    @staticmethod
    def from_r_to_0_1(val):
        return 1.0 / (1.0 + np.exp(-val))
