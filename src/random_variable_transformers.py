import numpy as np


class NTGTransform:
    @staticmethod
    def direct(x: np.array):
        return np.tan((x - 0.5) * np.pi)

    @staticmethod
    def direct_derivative(x: np.array):
        return np.pi / np.cos((x - 0.5) * np.pi) ** 2

    @staticmethod
    def reverse(x: np.array):
        return np.arctan(x) / np.pi + 0.5

    @staticmethod
    def reverse_derivative(x: np.array):
        return 1 / np.pi / (1 + x**2)


class IdentityTransform:
    @staticmethod
    def direct(x: np.array):
        return x

    @staticmethod
    def reverse(x: np.array):
        return x
