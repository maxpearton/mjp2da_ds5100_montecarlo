# Monte Carlo Simulator

## Metadata
- Author: Max Pearton
- Project: Monte Carlo Simulator

## Synopsis

### Create Dice
```python
from montecarlo import Die

# Example 1: Create a six-sided die with default weights
die1 = Die(['1', '2', '3', '4', '5', '6'])

# Example 2: Create a four-sided die with custom weights
die2 = Die(['A', 'B', 'C', 'D'], [0.1, 0.2, 0.3, 0.4])

Play a Game
from montecarlo import Die, Game

# Create dice for the game
dice = [Die(['A', 'B', 'C']), Die(['X', 'Y', 'Z'])]

# Create a game with the dice
game = Game(dice)

# Play the game 100 times
game.play(100)

# Show results in wide format
wide_results = game.show_results(format='wide')

# Show results in narrow format
narrow_results = game.show_results(format='narrow')

Analyze a Game
from montecarlo import Die, Game, Analyzer

# Create dice for the game
dice = [Die(['A', 'B', 'C']), Die(['X', 'Y', 'Z'])]

# Create a game with the dice
game = Game(dice)

# Play the game 100 times
game.play(100)

# Analyze the game
analyzer = Analyzer(game)

# Compute the number of jackpots
jackpot_count = analyzer.jackpot()

# Compute face counts per roll
face_counts = analyzer.face_counts_per_roll()

# Compute distinct combinations and their counts
combinations = analyzer.combo_count()

# Compute distinct permutations and their counts
permutations = analyzer.permutation_count()

#API Description
Die Class
Methods
__init__(faces, weights=None)

Initializes a Die object with faces and weights.
Parameters:
faces (list): An array representing distinct faces on the die.
weights (list, optional): An array representing weights corresponding to the faces. If None, all weights are set to 1.0.
_validate_faces_and_weights()

Validates that faces are distinct and weights are numeric.
_validate_face(face: str)

Validates a single face.
Parameters:
face (str): The face to be validated.
_validate_weight(weight)

Validates a single weight.
Parameters:
weight: The weight to be validated.
change_weight(face: str, weight: float)

Changes the weight of a specific face on the die.
Parameters:
face (str): The face for which the weight will be changed.
weight (float): The new weight value for the specified face.
roll(times: int = 1) -> List[str]

Rolls the die a specified number of times.
Parameters:
times (int, optional): The number of times the die will be rolled.
Returns:
A list of outcomes from the rolls.
show_state() -> pd.DataFrame

Returns a copy of the die's current state.
Returns:
A DataFrame containing the current weights of each face.
Game Class
Methods
__init__(dice: List[Die])

Initializes a Game object with a list of similar dice.
Parameters:
dice (list): A list of Die objects representing the dice in the game.
play(times: int)

Rolls all dice a specified number of times and saves results.
Parameters:
times (int): The number of times all dice will be rolled.
show_results(format: str = 'wide') -> pd.DataFrame

Returns a copy of the play results in a specified format.
Parameters:
format (str, optional): The format of the results. 'wide' or 'narrow'.
Returns:
A DataFrame containing the play results.
Analyzer Class
Methods
__init__(game: Game)

Initializes an Analyzer object with a Game for analysis.
Parameters:
game (Game): A Game object for analysis.
jackpot() -> int

Computes the number of jackpots in the game results.
Returns:
The number of jackpots.
face_counts_per_roll() -> pd.DataFrame

Computes face counts per roll in wide format.
Returns:
A DataFrame containing face counts for each roll.
combo_count() -> pd.DataFrame

Computes distinct combinations of faces and their counts.
Returns:
A DataFrame containing distinct combinations and their counts.
permutation_count() -> pd.DataFrame

Computes distinct permutations of faces and their counts.
Returns:
A DataFrame containing distinct permutations and their counts.
