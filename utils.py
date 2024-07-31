import math

TWOPI = 2.0 * math.pi
SQUTWOPI = math.sqrt(TWOPI)


def gaussian(x, mu, sig):
    return (
        1.0 / (SQUTWOPI * sig) * math.exp(-math.pow((x - mu) / sig, 2.0) / 2)
    )


class Gaussian:

    def __init__(self, mu, sig, peak=None):
        self.mu = mu
        self.sig = sig
        self.peak = peak or self.origin_peak
        self.s = self.peak / self.origin_peak

    @property
    def origin_peak(self):
        return 1/(SQUTWOPI*self.sig)

    def __call__(self, x):
        return (
                self.s / (SQUTWOPI * self.sig) * math.exp(-math.pow((x - self.mu) / self.sig, 2.0) / 2)
        )
