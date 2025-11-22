import numpy as np
import numpy.testing as npt
import unittest

from src.statistics_utils import StatisticsUtils


class TestStatisticsUtils(unittest.TestCase):
    """Test suite for StatisticsUtils class."""

    def test_example_moving_average_with_numpy_testing(self):
        """Ejemplo de test usando numpy.testing para comparar arrays de NumPy.

        Este test demuestra cómo usar numpy.testing.assert_allclose() para comparar
        arrays de NumPy con tolerancia para errores de punto flotante, lo cual es
        esencial cuando trabajamos con operaciones numéricas.
        """
        utils = StatisticsUtils()
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = utils.moving_average(arr, window=3)

        # Valores esperados para media móvil con window=3
        expected = np.array([2.0, 3.0, 4.0])

        # Usar numpy.testing.assert_allclose() para comparar arrays de NumPy
        # Esto maneja correctamente errores de punto flotante con tolerancia
        npt.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

    def test_example_min_max_scale_with_numpy_testing(self):
        """Ejemplo de test usando numpy.testing para verificar transformaciones numéricas.

        Este test demuestra cómo usar numpy.testing.assert_allclose() para verificar
        que una transformación numérica produce los resultados correctos en todo el array,
        considerando errores de punto flotante en cálculos matemáticos.
        """
        utils = StatisticsUtils()
        arr = [10.0, 20.0, 30.0, 40.0]
        result = utils.min_max_scale(arr)

        # Valores esperados después de min-max scaling: (x - min) / (max - min)
        # min=10, max=40, range=30
        # [10->0.0, 20->0.333..., 30->0.666..., 40->1.0]
        expected = np.array([0.0, 1 / 3, 2 / 3, 1.0])

        # Usar numpy.testing.assert_allclose() para comparar arrays numéricos
        # La tolerancia relativa y absoluta permite errores pequeños de punto flotante
        npt.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_moving_average_basic_case(self):
        """Test que verifica que el método moving_average calcula correctamente la media móvil
        de una secuencia numérica para un caso básico.

        Escenario esperado:
        - Crear una lista de números (ej: [1, 2, 3, 4])
        - Llamar a moving_average con window=2
        - Verificar que el resultado es correcto (ej: [1.5, 2.5, 3.5] para el array dado) (usar numpy.testing.assert_allclose() para comparar arrays de NumPy - esto es mejor que unittest porque maneja la comparación de arrays numéricos con tolerancia para errores de punto flotante)
        - Verificar que el resultado tiene la forma (shape) esperada (usar self.assertEqual para comparar tuplas de .shape - comparación simple, unittest es suficiente)
        """

        expected = np.array([1, 2, 3, 4])

        utils = StatisticsUtils()
        result = utils.moving_average(expected, window=2)

        npt.assert_allclose(result, np.array([1.5, 2.5, 3.5]), rtol=1e-7, atol=1e-7)
        self.assertEqual(result.shape, (3,))

    def test_moving_average_raises_for_invalid_window(self):
        """Test que verifica que el método moving_average lanza un ValueError cuando
        se proporciona una ventana (window) inválida.

        Escenario esperado:
        - Crear una lista de números (ej: [1, 2, 3])
        - Llamar a moving_average con window=0 (valor no positivo) y verificar que se lanza un ValueError (usar self.assertRaises)
        - Llamar a moving_average con window mayor que la longitud del array y verificar que se lanza un ValueError (usar self.assertRaises)
        """

        expected = np.array([1, 2, 3, 4])

        utils = StatisticsUtils()

        with self.assertRaises(ValueError):
            utils.moving_average(expected, window=0)

        with self.assertRaises(ValueError):
            utils.moving_average(expected, window=len(expected) + 1)

    def test_moving_average_only_accepts_1d_sequences(self):
        """Test que verifica que el método moving_average lanza un ValueError cuando
        se llama con una secuencia multidimensional.

        Escenario esperado:
        - Crear una secuencia bidimensional (ej: [[1, 2], [3, 4]])
        - Llamar a moving_average con esa secuencia y verificar que se lanza un ValueError indicando que solo se aceptan secuencias 1D (usar self.assertRaises)
        """

        expected = np.array([[1, 2], [3, 4]])

        utils = StatisticsUtils()

        with self.assertRaises(ValueError):
            utils.moving_average(expected, window=2)

    def test_zscore_has_mean_zero_and_unit_std(self):
        """Test que verifica que el método zscore calcula correctamente los z-scores
        de una secuencia numérica, comprobando que el resultado tiene media cero y
        desviación estándar unitaria.

        Escenario esperado:
        - Crear una lista de números (ej: [10, 20, 30, 40])
        - Llamar a zscore para obtener los z-scores (resultado es un array de NumPy)
        - Verificar que la media del resultado es aproximadamente 0 (usar self.assertAlmostEqual para un solo valor numérico - unittest es suficiente)
        - Verificar que la desviación estándar del resultado es aproximadamente 1 (usar self.assertAlmostEqual para un solo valor numérico - unittest es suficiente)
        """

        expected = np.array([10, 20, 30, 40])

        utils = StatisticsUtils()
        result = utils.zscore(expected)

        self.assertAlmostEqual(result.mean(), 0.0, places=7)
        self.assertAlmostEqual(result.std(), 1.0, places=7)

    def test_zscore_raises_for_zero_std(self):
        """Test que verifica que el método zscore lanza un ValueError cuando
        se llama con una secuencia que tiene desviación estándar cero
        (todos los valores son iguales).

        Escenario esperado:
        - Crear una lista con todos los valores iguales (ej: [5, 5, 5])
        - Llamar a zscore con esa secuencia y verificar que se lanza un ValueError indicando que la desviación estándar es cero (usar self.assertRaises)
        """

        expected = np.array([5, 5, 5])

        utils = StatisticsUtils()

        with self.assertRaises(ValueError):
            utils.zscore(expected)

    def test_min_max_scale_maps_to_zero_one_range(self):
        """Test que verifica que el método min_max_scale escala correctamente una secuencia
        numérica al rango [0, 1], donde el valor mínimo se mapea a 0 y el máximo a 1.

        Escenario esperado:
        - Crear una lista de números (ej: [2, 4, 6])
        - Llamar a min_max_scale para obtener los valores escalados (resultado es un array de NumPy)
        - Verificar que el valor mínimo del resultado es 0.0 (usar self.assertAlmostEqual para un solo valor numérico - unittest es suficiente)
        - Verificar que el valor máximo del resultado es 1.0 (usar self.assertAlmostEqual para un solo valor numérico - unittest es suficiente)
        - Verificar que los valores transformados son correctos (ej: [0.0, 0.5, 1.0] para [2, 4, 6]) (usar numpy.testing.assert_allclose() para comparar el array completo - esto es necesario para comparar arrays de NumPy con tolerancia para errores de punto flotante)
        """

        expected = np.array([2, 4, 6])

        utils = StatisticsUtils()
        result = utils.min_max_scale(expected)

        self.assertAlmostEqual(result.min(), 0.0, places=7)
        self.assertAlmostEqual(result.max(), 1.0, places=7)
        npt.assert_allclose(result, np.array([0.0, 0.5, 1.0]), rtol=1e-7, atol=1e-7)

    def test_min_max_scale_raises_for_constant_values(self):
        """Test que verifica que el método min_max_scale lanza un ValueError cuando
        se llama con una secuencia donde todos los valores son iguales (no hay variación).

        Escenario esperado:
        - Crear una lista con todos los valores iguales (ej: [3, 3, 3])
        - Llamar a min_max_scale con esa secuencia y verificar que se lanza un ValueError indicando que todos los valores son iguales (usar self.assertRaises)
        """

        expected = np.array([3, 3, 3])

        utils = StatisticsUtils()

        with self.assertRaises(ValueError):
            utils.min_max_scale(expected)


if __name__ == "__main__":
    unittest.main()
