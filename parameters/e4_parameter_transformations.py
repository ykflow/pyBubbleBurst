import numpy as np
from parameters.abstract_parameters import AbstractParameterTransformer


class E4ParameterTransformations(AbstractParameterTransformer):

    def __init__(self):
        # Explicit registration order for flat vector index mapping
        self.param_names = ["delta", "beta", "gamma", "omega", "alpha", "kappa", "c", "sigma2"]
        self._k_unknown = len(self.param_names)

        self._idx_delta = 0
        self._idx_beta = 1
        self._idx_gamma = 2
        self._idx_omega = 3
        self._idx_alpha = 4
        self._idx_kappa = 5
        self._idx_c = 6
        self._idx_sigma2 = 7

        self.continuous_names = ["sigma2"]
        self.discontinuous_names = ["delta", "beta", "gamma", "omega", "alpha", "kappa", "c"]

        self._idx_discont = np.array([self.param_names.index(name) for name in self.discontinuous_names], dtype=np.intp)
        self._idx_cont = np.array([self.param_names.index(name) for name in self.continuous_names], dtype=np.intp)

    def split_vector(self, full_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Splits a full flat vector into discontinuous and continuous sub-vectors.

        Can be used on either transformed or untransformed vectors.
        """
        return full_vector[self._idx_discont], full_vector[self._idx_cont]

    def merge_params(self, discontinuous_vector: np.ndarray, continuous_vector: np.ndarray) -> np.ndarray:
        """Reconstructs the full flat vector from isolated discontinuous and continuous parts."""
        full_vector = np.empty(self._k_unknown, dtype=np.float64)
        full_vector[self._idx_discont] = discontinuous_vector
        full_vector[self._idx_cont] = continuous_vector
        return full_vector

    def transform(self, untransformed_vector: np.ndarray) -> np.ndarray:
        """Maps standard parameter vectors to unconstrained R spaces."""
        transformed = np.zeros(self._k_unknown, dtype=np.float64)
        transformed[self._idx_delta] = untransformed_vector[self._idx_delta]
        transformed[self._idx_c] = untransformed_vector[self._idx_c]
        transformed[self._idx_beta] = self.from_0_1_to_r(untransformed_vector[self._idx_beta])
        transformed[self._idx_alpha] = self.from_1_plus_to_r(untransformed_vector[self._idx_alpha])
        transformed[self._idx_gamma] = self.from_pos_to_r(untransformed_vector[self._idx_gamma])
        transformed[self._idx_omega] = self.from_pos_to_r(untransformed_vector[self._idx_omega])
        transformed[self._idx_kappa] = self.from_pos_to_r(untransformed_vector[self._idx_kappa])
        transformed[self._idx_sigma2] = self.from_pos_to_r(untransformed_vector[self._idx_sigma2])
        return transformed

    def untransform(self, transformed_vector: np.ndarray) -> np.ndarray:
        """Maps unconstrained R optimizer spaces back to valid model domains."""
        untransformed = np.zeros(self._k_unknown, dtype=np.float64)
        untransformed[self._idx_delta] = transformed_vector[self._idx_delta]
        untransformed[self._idx_c] = transformed_vector[self._idx_c]
        untransformed[self._idx_beta] = self.from_r_to_0_1(transformed_vector[self._idx_beta])
        untransformed[self._idx_alpha] = self.from_r_to_1_plus(transformed_vector[self._idx_alpha])
        untransformed[self._idx_gamma] = self.from_r_to_pos(transformed_vector[self._idx_gamma])
        untransformed[self._idx_omega] = self.from_r_to_pos(transformed_vector[self._idx_omega])
        untransformed[self._idx_kappa] = self.from_r_to_pos(transformed_vector[self._idx_kappa])
        untransformed[self._idx_sigma2] = self.from_r_to_pos(transformed_vector[self._idx_sigma2])
        return untransformed

    def to_kwargs(self, untransformed_vector: np.ndarray) -> dict:
        return dict(zip(self.param_names, untransformed_vector))

    def get_mle_inits(self) -> np.ndarray:
        inits = np.empty(self._k_unknown, dtype=np.float64)
        inits[self._idx_delta] = 0.1
        inits[self._idx_beta] = 0.95
        inits[self._idx_gamma] = 0.7
        inits[self._idx_omega] = 0.2
        inits[self._idx_alpha] = 1.03
        inits[self._idx_kappa] = 7
        inits[self._idx_c] = -0.1
        inits[self._idx_sigma2] = 1
        return self.transform(inits) *0.9