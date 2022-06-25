import numpy as np
import itertools


class RadialBasisFunctionExtractor:
    def __init__(self, number_of_kernels_per_dim):
        lower = -2.
        upper = 2.
        mus_per_dim = []
        for number_of_kernels in number_of_kernels_per_dim:
            direction = (upper - lower) / (number_of_kernels - 1)
            mus = [lower + i * direction for i in range(number_of_kernels)]
            mus_per_dim.append(mus)
        self.mus = [np.array(mu) for mu in itertools.product(*mus_per_dim)]
        self.sigmas = [2. / (min(number_of_kernels_per_dim) - 1)] * len(self.mus)

    @staticmethod
    def _compute_kernel(states, mu, sigma):
        exponent = np.linalg.norm(states - mu, axis=1)
        exponent = -exponent / (2 * np.square(sigma))
        return np.exp(exponent)

    def encode_states_with_radial_basis_functions(self, states):
        features = [
            np.expand_dims(self._compute_kernel(states, self.mus[i], self.sigmas[i]), axis=1)
            for i in range(self.get_number_of_features())
        ]
        return np.concatenate(features, axis=1)

    def get_number_of_features(self):
        return len(self.mus)
