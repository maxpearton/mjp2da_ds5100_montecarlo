# montecarlo/montecarlo.py
import numpy as np
import pandas as pd
from typing import List

class Die:
    def __init__(self, faces, weights=None):
        self.faces = np.array(faces)
        self.weights = np.array(weights) if weights is not None else np.ones_like(self.faces, dtype=float)
        self._validate_faces_and_weights()
        self._die_state = pd.DataFrame({'weights': self.weights}, index=self.faces)

    def _validate_faces_and_weights(self):
        # Validate that faces are distinct
        if len(set(self.faces)) != len(self.faces):
            raise ValueError("Faces must be distinct.")

        # Validate that weights are numeric
        if not np.issubdtype(self.weights.dtype, np.number):
            raise TypeError("Weights must be numeric.")

    def _validate_face(self, face):
        if face not in self.faces:
            raise IndexError(f"Invalid face '{face}'.")

    def _validate_weight(self, weight):
        if not isinstance(weight, (int, float)) or np.isnan(weight):
            raise TypeError("Weight must be a numeric value.")
    def change_weight(self, face: str, weight: float):
        if face not in self.faces:
            raise IndexError("Invalid face value.")
        if not np.issubdtype(type(weight), np.number):
            raise TypeError("Weight must be numeric.")
        self.weights[self.faces == face] = weight

    def roll(self, times: int = 1) -> List[str]:
        outcomes = np.random.choice(self.faces, times, p=self.weights / np.sum(self.weights))
        return outcomes.tolist()

    def show_state(self) -> pd.DataFrame:
        return self._die_state.copy()
class Game:
    """
    Game class for simulating the rolling of multiple similar dice.

    Attributes:
    - dice (List[Die]): List of Die objects representing the dice in the game.
    - play_results (pd.DataFrame): DataFrame to store results of the most recent play.

    Methods:
    - __init__(dice: List[Die]): Initializes a Game object with a list of similar dice.
    - play(times: int): Rolls all dice a specified number of times and saves results.
    - show_results(format: str = 'wide') -> pd.DataFrame: Returns a copy of the play results.
    """
    def __init__(self, dice: List[Die]):
        self.dice = dice
        self.play_results = None

    def play(self, times: int):
        outcomes = {f'Die_{i+1}': die.roll(times) for i, die in enumerate(self.dice)}
        self.play_results = pd.DataFrame(outcomes)

    def show_results(self, format: str = 'wide') -> pd.DataFrame:
        if format == 'wide':
            return self.play_results.copy()
        elif format == 'narrow':
            return pd.melt(self.play_results, var_name='Die', value_name='Outcome')
        else:
            raise ValueError("Invalid format. Use 'wide' or 'narrow'.")

class Analyzer:
    """
    Analyzer class for computing statistical properties of a Game.

    Attributes:
    - game (Game): Game object for analysis.

    Methods:
    - __init__(game: Game): Initializes an Analyzer object with a Game for analysis.
    - jackpot() -> int: Computes the number of jackpots in the game results.
    - face_counts_per_roll() -> pd.DataFrame: Computes face counts per roll in wide format.
    - combo_count() -> pd.DataFrame: Computes distinct combinations of faces and their counts.
    - permutation_count() -> pd.DataFrame: Computes distinct permutations of faces and their counts.
    """
    def __init__(self, game: Game):
        self.game = game

    def jackpot(self) -> int:
        results = self.game.show_results()
        print("Results:")
        print(results)

        jackpot_count = (results.apply(lambda row: row.nunique() == 1, axis=1)).sum()
        print("Jackpot Count:")
        print(jackpot_count)

        return jackpot_count

    def face_counts_per_roll(self) -> pd.DataFrame:
        return self.game.show_results().apply(lambda col: col.value_counts())

    def combo_count(self) -> pd.DataFrame:
        combinations = self.game.show_results().apply(tuple, axis=1)
        return pd.DataFrame(combinations.value_counts(), columns=['Count'])

    def permutation_count(self) -> pd.DataFrame:
        permutations = self.game.show_results().apply(tuple, axis=1)
        return pd.DataFrame(permutations.value_counts(), columns=['Count'])