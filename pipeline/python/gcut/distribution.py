import os
import json
import numpy as np
from pipeline_util import timer


class Distribution:
    def __init__(self):
        self.pdf, self.cdf = None, None
        # x -> (x - self.mean) / self.std
        self._mean, self._std = None, None
        # Integral_{scale(0), scale(pi)}(polynomial(self.coefficients)) =
        # self.normalizer
        # _polynomial integrate to _normalizer on [scale(0), scale(pi)]
        self._polynomial, self._normalizer = None, None

    def clear(self):
        self.pdf, self.cdf = None, None
        self._mean, self._std = None, None
        self._normalizer = None

    # conditional: None or 'mouse'
    # precedence: conditional > brain_regions > species > cell type
    # if conditional = 'mouse', species is ignored
    @timer
    def load_distribution(self, conditional='mouse', brain_region='neocortex',
                          species='mouse', cell_type='principal neuron'):
        if conditional == 'mouse':
            json_name = 'gof_mouse_distributions.json'
        else:
            json_name = 'gof_distributions.json'
        json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), json_name)
        with open(json_path, 'r') as f:
            all_parameters = json.load(f)
        if conditional == 'mouse':
            key = brain_region if brain_region is not None else cell_type
        else:
            if brain_region is not None:
                key = brain_region
            elif species is not None:
                key = species
            else:
                key = cell_type
        if key is None:
            key = 'neocortex'
        if key not in all_parameters:
            raise ValueError('{} not found in {}'.format(key, json_name))
        self._polynomial = np.poly1d(all_parameters[key]['coefficients'])
        self._mean = all_parameters[key]['mean']
        self._std = all_parameters[key]['std']
        self.create_cdf()

    # analytically derive cdf from pdf
    def create_cdf(self):
        # highest order to 0
        poly_coefficients = self._polynomial.c
        orders = [order for order in range(len(poly_coefficients) - 1, -1, -1)]
        # integrate the polynomial to find the normalizer
        integral_coefficients = []
        for poly_coefficient, order in zip(poly_coefficients, orders):
            integral_coefficients.append(poly_coefficient / (order + 1))
        integral_coefficients.append(0)
        integral = np.poly1d(integral_coefficients)
        # I_{a, b}f(x) = F(b) - F(a), F'(x) = f(x)
        self._normalizer = integral(self.scale(np.pi)) - integral(self.scale(0))
        # noramlize the polynomial to pdf
        self.pdf = self._polynomial / self._normalizer
        # integrate the pdf to get cdf + constant
        pdf_integral = integral / self._normalizer
        constant = -pdf_integral(self.scale(0))
        cdf_coefficients = pdf_integral.c
        cdf_coefficients[-1] = constant
        self.cdf = np.poly1d(cdf_coefficients)
        print('cdf: cdf(0) = {}, cdf(pi) = {}'.format(self.cdf(self.scale(0)),
                                                      self.cdf(self.scale(np.pi))))

    def scale(self, val):
        return (val - self._mean) / self._std

    # P(X >= theta)
    def probability(self, theta):
        assert 0 <= theta <= np.pi
        P = 1 - self.cdf(self.scale(theta))
        assert 0 <= P <= 1
        return P

