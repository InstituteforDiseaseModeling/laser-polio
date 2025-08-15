import numba as nb
import numpy as np

# Add common distributions so they can be imported directly; assigned to a variable since used in help messages
dist_list = [
    "constant",
    "exponential",
    "gamma",
    "lognormal",
    "normal",
    "poisson",
    "uniform",
]
__all__ = list(dist_list)


class Distribution:
    def __init__(self, dist_type, **pars):
        """
        Initializes the distribution.

        :param dist_type: Type of distribution (e.g., "poisson", "uniform", "gamma", etc.).
        :param pars: Parameters specific to the chosen distribution.
        """
        self.dist_type = dist_type
        self.pars = pars

        # Validate the distribution type
        supported_distributions = {
            "constant",
            "exponential",
            "gamma",
            "lognormal",
            "normal",
            "poisson",
            "uniform",
        }
        if dist_type not in supported_distributions:
            raise ValueError(f"Unsupported distribution: {dist_type}. Supported: {supported_distributions}")

    def sample(self, size=1):
        """
        Generates random samples from the specified distribution.

        :param size: Number of samples to generate.
        :return: A NumPy array of sampled values.
        """
        dist = self.dist_type
        if dist == "constant":
            return np.full(size, self.pars.get("value", 1))
        elif dist == "exponential":
            return np.random.exponential(self.pars.get("scale", 1.0), size)
        elif dist == "gamma":
            # return np.random.gamma(self.pars.get("shape", 2.0), self.pars.get("scale", 1.0), size)
            out = np.empty(size, np.float64)
            nb_gamma(
                self.pars.get("shape", 2.0),
                self.pars.get("scale", 1.0),
                size,
                out,
            )
            return out
        elif dist in ["lognormal", "lognormal_int"]:
            par1 = self.pars.get("mean", 1.0)
            par2 = self.pars.get("sigma", 0.5)
            if par1 > 0:
                mean = np.log(par1**2 / np.sqrt(par2**2 + par1**2))  # Computes the mean of the underlying normal distribution
                sigma = np.sqrt(np.log(par2**2 / par1**2 + 1))  # Computes sigma for the underlying normal distribution
                samples = np.random.lognormal(mean=mean, sigma=sigma, size=size)
            else:
                samples = np.zeros(size)
            if self.dist_type == "lognormal_int":
                samples = np.round(samples)
            return samples
        elif dist == "normal":
            # return np.random.normal(self.pars.get("mean", 0.0), self.pars.get("std", 1.0), size)
            out = np.empty(size, np.float64)
            nb_normal(
                self.pars.get("mean", 0.0),
                self.pars.get("std", 1.0),
                size,
                out,
            )
            return out
        elif dist == "poisson":
            return np.random.poisson(self.pars.get("lam", 5), size)
        elif dist == "uniform":
            return np.random.randint(self.pars.get("min", 2), self.pars.get("max", 10), size)
        else:
            raise ValueError(f"Unsupported distribution type: {dist}")

    def __call__(self, size=1):
        """
        Allows the instance to be called like a function to sample values.

        :param size: Number of samples to generate.
        :return: A NumPy array of sampled values.
        """
        return self.sample(size)

    def __repr__(self):
        return f"Distribution(type={self.dist_type}, pars={self.pars})"


# Define helper functions for common distributions
def constant(value):
    return Distribution("constant", value=value)


def exponential(scale):
    return Distribution("exponential", scale=scale)


def gamma(shape, scale):
    return Distribution("gamma", shape=shape, scale=scale)


def lognormal(mean, sigma):
    return Distribution("lognormal", mean=mean, sigma=sigma)


def lognormal_int(mean, sigma):
    return Distribution("lognormal_int", mean=mean, sigma=sigma)


def normal(mean, std):
    return Distribution("normal", mean=mean, std=std)


def poisson(lam):
    return Distribution("poisson", lam=lam)


def uniform(min, max):
    return Distribution("uniform", min=min, max=max)


@nb.njit((nb.float64, nb.float64, nb.int32, nb.float64[:]), parallel=True)
def nb_normal(mean, std, size, dest):
    for i in nb.prange(size):
        dest[i] = np.random.normal(mean, std)

    return


@nb.njit((nb.float64, nb.float64, nb.int32, nb.float64[:]), parallel=True, cache=True)
def nb_gamma(shape, scale, size, dest):
    for i in nb.prange(size):
        dest[i] = np.random.gamma(shape, scale)

    return
