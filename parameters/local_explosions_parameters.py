import numpy as np
from parameters.abstract_parameters import AbstractParameterTransformer


class LocalExplosionsParameters(AbstractParameterTransformer):

    def __init__(self):
        # Explicit registration order for flat vector index mapping
        self.param_names = [ "delta", "beta", "gamma", "omega", "alpha",  "kappa", "c", "sigma2"]
        self._k_unknown = len(self.param_names)

        # Build index positions
        self._idx_delta = 0
        self._idx_beta = 1
        self._idx_gamma = 2
        self._idx_omega = 3
        self._idx_alpha = 4
        self._idx_kappa = 5
        self._idx_c = 6
        self._idx_sigma2 = 7

    def transform(self, untransformed_vector: np.ndarray) -> np.ndarray:
        """Maps standard parameter vectors to unconstrained R spaces."""
        transformed = np.zeros(self._k_unknown, dtype=np.float64)

        # Unconstrained completely (Identical mappings)
        transformed[self._idx_delta] = untransformed_vector[self._idx_delta]
        transformed[self._idx_c] = untransformed_vector[self._idx_c]

        # Bounded between (0, 1) -> Logit transform
        transformed[self._idx_beta] = self.from_0_1_to_r(untransformed_vector[self._idx_beta])

        # Bounded strictly positive (0, inf) -> Log transform
        transformed[self._idx_alpha] = self.from_1_plus_to_r(untransformed_vector[self._idx_alpha])
        transformed[self._idx_gamma] = self.from_pos_to_r(untransformed_vector[self._idx_gamma])
        transformed[self._idx_omega] = self.from_pos_to_r(untransformed_vector[self._idx_omega])
        transformed[self._idx_kappa] = self.from_pos_to_r(untransformed_vector[self._idx_kappa])
        transformed[self._idx_sigma2] = self.from_pos_to_r(untransformed_vector[self._idx_sigma2])

        return transformed

    def untransform(self, transformed_vector: np.ndarray) -> np.ndarray:
        """Maps unconstrained R optimizer spaces back to valid model domains."""
        untransformed = np.zeros(self._k_unknown, dtype=np.float64)

        # Re-convert identities
        untransformed[self._idx_delta] = transformed_vector[self._idx_delta]
        untransformed[self._idx_c] = transformed_vector[self._idx_c]

        # Re-convert Logit bounds to stable (0, 1) spaces
        untransformed[self._idx_beta] = self.from_r_to_0_1(transformed_vector[self._idx_beta])

        # Re-convert Log bounds back to strictly positive scalars
        untransformed[self._idx_alpha] = self.from_r_to_1_plus(transformed_vector[self._idx_alpha])
        untransformed[self._idx_gamma] = self.from_r_to_pos(transformed_vector[self._idx_gamma])
        untransformed[self._idx_omega] = self.from_r_to_pos(transformed_vector[self._idx_omega])
        untransformed[self._idx_kappa] = self.from_r_to_pos(transformed_vector[self._idx_kappa])
        untransformed[self._idx_sigma2] = self.from_r_to_pos(transformed_vector[self._idx_sigma2])

        return untransformed

    def to_kwargs(self, untransformed_vector: np.ndarray) -> dict:
        """Converts raw, valid parameters into explicit Numba filter arguments."""
        return dict(zip(self.param_names, untransformed_vector))

    def get_mle_inits(self) -> np.ndarray:
        """Generates the starting position array already projected into R space."""
        # Baseline guesses in valid constrained fields
        inits = np.zeros(self._k_unknown, dtype=np.float64)
        inits[self._idx_delta] = 0.1
        inits[self._idx_beta] = 0.85
        inits[self._idx_gamma] = 0.7
        inits[self._idx_omega] = 0.2
        inits[self._idx_alpha] = 1.1
        inits[self._idx_kappa] = 1
        inits[self._idx_c] = -0.1
        inits[self._idx_sigma2] = 0.2

        # Return them wrapped safely inside unconstrained spaces
        return self.transform(inits)
