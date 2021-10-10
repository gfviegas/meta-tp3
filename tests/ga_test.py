from meta_tp3.ga import Individual
from tests import *
from unittest import TestCase

import pytest


class TestGA(TestCase):
    def test_individual_default_values(self):
        ind = Individual()
        self.assertEqual(ind.value, [])
        self.assertEqual(ind.fitness, None)
        self.assertEqual(ind.generation, 0)

    def test_individual_creation_with_value(self):
        ind = Individual([1, 0.34])
        self.assertEqual(ind.value, [1, 0.34])
        self.assertEqual(ind.fitness, None)
        self.assertEqual(ind.generation, 0)

    def test_individual_creation_with_generation(self):
        ind = Individual(generation=4)
        self.assertEqual(ind.value, [])
        self.assertEqual(ind.fitness, None)
        self.assertEqual(ind.generation, 4)
