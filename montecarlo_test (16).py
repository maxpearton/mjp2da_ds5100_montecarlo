import unittest
import numpy as np
import pandas as pd
from montecarlo import Die, Game, Analyzer

class TestMonteCarlo(unittest.TestCase):
    def test_die_initialization(self):
        faces = np.array(['A', 'B', 'C'])  # Example distinct faces
        die = Die(faces)
    
    # Check that faces are distinct
        self.assertEqual(len(set(die.faces)), len(die.faces))
    
    # Check that weights are all ones
        expected_weights = np.ones_like(die.faces, dtype=float)
        np.testing.assert_array_equal(die.weights, expected_weights)

    def test_die_validate_face(self):
        faces = np.array(['A', 'B', 'C'])
        die = Die(faces)

        # Valid face should not raise an error
        self.assertIsNone(die._validate_face('A'))

        # Invalid face should raise an error
        with self.assertRaises(IndexError):
            die._validate_face('D')

    def test_die_validate_weight(self):
        faces = np.array(['A', 'B', 'C'])
        die = Die(faces)

        # Valid weight should not raise an error
        self.assertIsNone(die._validate_weight(2.0))

        # Invalid weight should raise an error
        with self.assertRaises(TypeError):
            die._validate_weight('invalid')
            
    def test_die_validate_faces_and_weights_distinct_faces(self):
    # Valid faces (distinct) and weights should not raise an error
        faces = np.array(['A', 'B', 'C'])
        weights = np.array([0.2, 0.3, 0.5])
        die = Die(faces, weights)
        self.assertIsNone(die._validate_faces_and_weights())

    # Invalid faces (not distinct) should raise an error
        faces_not_distinct = np.array(['A', 'B', 'A'])
        die_not_distinct = Die(faces_not_distinct, weights)
    
        try:
            die_not_distinct._validate_faces_and_weights()
        except ValueError as e:
            print(f"Expected ValueError: {e}")
            return  # Add this line to exit the test after printing the error

    # If no error occurred, raise an AssertionError
        self.fail("Expected ValueError but no error was raised")

    def test_die_validate_faces_and_weights_numeric_weights(self):
        # Valid faces and weights (numeric) should not raise an error
        faces = np.array(['A', 'B', 'C'])
        weights = np.array([0.2, 0.3, 0.5])
        die = Die(faces, weights)
        self.assertIsNone(die._validate_faces_and_weights())

        # Invalid weights (non-numeric) should raise an error
        faces = np.array(['A', 'B', 'C'])
        weights_non_numeric = np.array(['a', 'b', 'c'])
        die_non_numeric = Die(faces, weights_non_numeric)
        with self.assertRaises(TypeError):
            die_non_numeric._validate_faces_and_weights()
            
    def test_die_change_weight(self):
        faces = np.array(['A', 'B', 'C'])
        die = Die(faces)
        die.change_weight('A', 2.0)
        self.assertEqual(die.weights[0], 2.0)

    def test_die_roll(self):
        faces = np.array(['A', 'B', 'C'])
        die = Die(faces)
        outcomes = die.roll(10)
        self.assertEqual(len(outcomes), 10)
        
    def test_die_show_state(self):
        faces = np.array(['A', 'B', 'C'])
        weights = np.array([0.2, 0.3, 0.5])
        die = Die(faces, weights)

        # Check if the returned result is a DataFrame
        state = die.show_state()
        self.assertIsInstance(state, pd.DataFrame)   

    def test_game_initialization(self):
        die1 = Die(np.array(['A', 'B', 'C']))
        die2 = Die(np.array(['X', 'Y', 'Z']))
        game = Game([die1, die2])
        self.assertEqual(game.dice, [die1, die2])

    def test_game_play(self):
        die1 = Die(np.array(['A', 'B', 'C']))
        die2 = Die(np.array(['X', 'Y', 'Z']))
        game = Game([die1, die2])
        game.play(5)
        self.assertEqual(len(game.show_results()), 5)

    def test_game_show_results_wide_format(self):
        die1 = Die(np.array(['A', 'B', 'C']))
        die2 = Die(np.array(['X', 'Y', 'Z']))
        game = Game([die1, die2])
        game.play(3)
        results = game.show_results(format='wide')

        # Check if the returned results is a DataFrame
        self.assertIsInstance(results, pd.DataFrame)
        
    def test_analyzer_initialization(self):
        die1 = Die(np.array(['A', 'B', 'C']))
        die2 = Die(np.array(['X', 'Y', 'Z']))
        game = Game([die1, die2])
        analyzer = Analyzer(game)
        self.assertEqual(analyzer.game, game)

    def test_analyzer_jackpot(self):
        die1 = Die(np.array(['A', 'B', 'C']))
        die2 = Die(np.array(['X', 'Y', 'Z']))
        game = Game([die1, die2])
        analyzer = Analyzer(game)

    # Call the play method to generate results
        game.play(5)

    # Now you can safely access show_results
        for index, row in analyzer.game.show_results().iterrows():
            print(f"Row {index}: {row.tolist()}")

        self.assertEqual(analyzer.jackpot(), 0)

    def test_analyzer_face_counts_per_roll(self):
        die1 = Die(np.array(['A', 'B', 'C']))
        die2 = Die(np.array(['X', 'Y', 'Z']))
        game = Game([die1, die2])
        game.play(3)
        analyzer = Analyzer(game)
        face_counts = analyzer.face_counts_per_roll()
        self.assertIsInstance(face_counts, pd.DataFrame)

    def test_analyzer_combo_count(self):
        die1 = Die(np.array(['A', 'B', 'C']))
        die2 = Die(np.array(['X', 'Y', 'Z']))
        game = Game([die1, die2])
        game.play(3)
        analyzer = Analyzer(game)
        combos = analyzer.combo_count()
        self.assertIsInstance(combos, pd.DataFrame)

    def test_analyzer_permutation_count(self):
        die1 = Die(np.array(['A', 'B', 'C']))
        die2 = Die(np.array(['X', 'Y', 'Z']))
        game = Game([die1, die2])
        game.play(3)
        analyzer = Analyzer(game)
        perms = analyzer.permutation_count()
        self.assertIsInstance(perms, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()