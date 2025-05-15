import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class YahtzeeEnv(gym.Env):
    """
    A custom Yahtzee environment that follows the OpenAI Gym interface.
    Modified to work better with standard RL algorithms by using discrete spaces
    and providing a flattened action space.
    """
    
    def __init__(self):
        super(YahtzeeEnv, self).__init__()
        
        # Define scoring categories and their maximum scores
        self.categories = {
            'ones': 5,         # Max score: 5 (1×5)
            'twos': 10,        # Max score: 10 (2×5)
            'threes': 15,      # Max score: 15 (3×5)
            'fours': 20,       # Max score: 20 (4×5)
            'fives': 25,       # Max score: 25 (5×5)
            'sixes': 30,       # Max score: 30 (6×5)
            'three_of_a_kind': 30,
            'four_of_a_kind': 30,
            'full_house': 25,
            'small_straight': 30,
            'large_straight': 40,
            'yahtzee': 50,
            'chance': 30
        }
        
        # Flatten action space into a single discrete space
        # Actions 0-12: Choose scoring category
        # Actions 13-44: Reroll combinations (2^5 = 32 possible reroll combinations)
        self.action_space = spaces.Discrete(45)
        
        # Define observation space using MultiDiscrete
        self.observation_space = spaces.MultiDiscrete([
            6, 6, 6, 6, 6,  # dice values (0-5 representing 1-6)
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # available categories (binary) (6-18) (13 categories)
            3  # remaining rolls (0,1,2) (19)
        ])
        
        # Initialize game state
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.dice = self._roll_dice()
        self.remaining_rolls = 2
        self.available_categories = np.ones(13, dtype=np.int8)
        self.scores = np.zeros(13, dtype=np.int32)
        self.total_score = 0
        self.is_bonus = False
        
        return self._get_observation(), {}
    
    def _roll_dice(self, reroll_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Roll the dice according to reroll mask."""
        if reroll_mask is None:
            return np.random.randint(0, 6, size=5)  # 0-5 representing 1-6
        
        new_dice = self.dice.copy()
        for i, reroll in enumerate(reroll_mask):
            if reroll:
                new_dice[i] = np.random.randint(0, 6)
        return new_dice
    
    def _decode_action(self, action: int) -> Tuple[int, np.ndarray]:
        """Convert flat action space to category and reroll mask."""
        if action < 13:  # Scoring actions
            return action, np.zeros(5, dtype=np.int8)
        else:  # Reroll actions
            reroll_idx = action - 13
            return -1, np.array([int(x) for x in format(reroll_idx, '05b')])
    
    def _check_bonus(self) -> int:
        """Check if bonus is earned and return bonus score."""
        upper_section_score = np.sum(self.scores[:6])
        if not self.is_bonus and upper_section_score >= 63:
            self.is_bonus = True
            return 35
        return 0
    
    def _calculate_score(self, category_idx: int, dice: np.ndarray) -> int:
        """Calculate score for given category and dice combination."""
        dice = dice + 1  # Convert from 0-5 to 1-6
        dice_counts = np.bincount(dice, minlength=7)
        category_name = list(self.categories.keys())[category_idx]
        
        if category_name in ['ones', 'twos', 'threes', 'fours', 'fives', 'sixes']:
            number = category_idx + 1
            return (number * dice_counts[number]) + self._check_bonus()
        
        elif category_name == 'three_of_a_kind':
            if np.any(dice_counts >= 3):
                return np.sum(dice)
            return 0
        
        elif category_name == 'four_of_a_kind':
            if np.any(dice_counts >= 4):
                return np.sum(dice)
            return 0
        
        elif category_name == 'full_house':
            if np.any(dice_counts == 3) and np.any(dice_counts == 2):
                return 25
            return 0
        
        elif category_name == 'small_straight':
            for straight in [(1,2,3,4), (2,3,4,5), (3,4,5,6)]:
                if all(dice_counts[s] >= 1 for s in straight):
                    return 30
            return 0
        
        elif category_name == 'large_straight':
            if (all(dice_counts[1:7] == 1) or all(dice_counts[2:8] == 1)) or \
               (all(dice_counts[1:6] == 1) or all(dice_counts[2:7] == 1)):
                return 40
            return 0
        
        elif category_name == 'yahtzee':
            if np.any(dice_counts == 5):
                return 50
            return 0
        
        elif category_name == 'chance':
            return np.sum(dice)
        
        return 0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment using the given action."""
        category, reroll = self._decode_action(action)
        info = {}
                
        # Handle reroll action
        if category == -1:
            if self.remaining_rolls > 0:
                self.dice = self._roll_dice(reroll)
                self.remaining_rolls -= 1
                reward = 0  # Neutral reward for rerolling
                return self._get_observation(), reward, False, False, info
                
            else:
                reward = -50  # Penalty for invalid reroll
                return self._get_observation(), reward, False, False, info
        
        # Handle scoring action
        if not self.available_categories[category]:
            return self._get_observation(), -50, False, False, {'error': 'Category already used'}
        
        # Calculate score and update state
        score = self._calculate_score(category, self.dice)
        self.scores[category] = score
        self.available_categories[category] = 0
        self.total_score += score
        
        # Reset dice and rolls for next turn
        self.dice = self._roll_dice()
        self.remaining_rolls = 2
        
        # Check if game is done
        done = np.sum(self.available_categories) == 0
        
        # Calculate reward (use the score as the reward)
        reward = score
        
        return self._get_observation(), reward, done, False, {
            'total_score': self.total_score,
            'scores': self.scores.copy()
        }
    
    def _get_observation(self) -> np.ndarray:
        """Return current observation of the environment."""
        return np.concatenate([
            self.dice,  # 5 dice values (0-5)
            self.available_categories,  # 13 binary values
            [self.remaining_rolls]  # 1 value (0-2)
        ])
    
    def render(self, mode='human'):
        """Render the current state of the game."""
        if mode == 'human':
            print("\nCurrent Dice:", self.dice + 1)  # Convert back to 1-6 for display
            print("Remaining Rolls:", self.remaining_rolls)
            print("\nAvailable Categories:")
            for i, (category, available) in enumerate(zip(self.categories.keys(), self.available_categories)):
                if available:
                    possible_score = self._calculate_score(i, self.dice)
                    print(f"{category}: {possible_score} points possible")
            print("\nScored Categories:")
            for i, (category, score) in enumerate(zip(self.categories.keys(), self.scores)):
                if not self.available_categories[i]:
                    print(f"{category}: {score}")
            print("\nTotal Score:", self.total_score)
            if self.is_bonus:
                print("Bonus achieved: +35 points")
    
    def get_legal_actions(self) -> np.ndarray:
        """
        Return an indicator (binary) vector of legal actions.
        
        - For scoring actions (0-12): legal if that category is still available.
        - For reroll actions (13-44): legal if remaining_rolls > 0 and at least one die is rerolled.
        """
        legal = np.zeros(self.action_space.n, dtype=np.int8)
        # Scoring actions: allowed only if the category hasn't been used yet.
        for i in range(13):
            if self.available_categories[i]:
                legal[i] = 1
        # Reroll actions: allowed only if there are remaining rolls.
        if self.remaining_rolls > 0:
            for action in range(13, 45):
                # _, reroll = self._decode_action(action)
                # if np.any(reroll):  # Must reroll at least one die.
                legal[action] = 1
        return legal