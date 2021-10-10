from meta_tp3.ga import Individual
from meta_tp3.selection import tournament
from tests import *
from unittest import TestCase

import pytest


class TestGA(TestCase):
    def test_tournament_returns_empty_if_population_empty(self):
        # ind = Individual()
        pop = []
        tourn_result = tournament([], 1, 1)
        self.assertEqual(tourn_result, [])

    def test_tournament_raises_if_tourn_size_greater_than_pop_size(self):
        # ind = Individual()
        pop = [Individual(), Individual(), Individual()]
        with self.assertRaises(ValueError) as e:
            tournament(pop, 1, 4)
            self.assertEqual(str(e), "Sample larger than population or is negative")

    def test_tournament_should_return_best_fit_individual(self):
        pop = [Individual(), Individual()]
        pop[0].fitness = 50
        pop[1].fitness = 1

        tourn_result = tournament(pop, 1, 2)
        self.assertEqual(tourn_result[0], pop[0])
