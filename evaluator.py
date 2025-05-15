import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from YahtzeeEnv import YahtzeeEnv 
import gymnasium as gym


class HardCodedStrategy:
    def __init__(self):
        self.category_scores = {
            'ones': {'type': 'upper', 'number': 1},
            'twos': {'type': 'upper', 'number': 2},
            'threes': {'type': 'upper', 'number': 3},
            'fours': {'type': 'upper', 'number': 4},
            'fives': {'type': 'upper', 'number': 5},
            'sixes': {'type': 'upper', 'number': 6},
            'three_of_a_kind': {'type': 'three_kind'},
            'four_of_a_kind': {'type': 'four_kind'},
            'full_house': {'type': 'full_house'},
            'small_straight': {'type': 'small_straight'},
            'large_straight': {'type': 'large_straight'},
            'yahtzee': {'type': 'yahtzee'},
            'chance': {'type': 'chance'},
        }
    
    def calculate_reroll_strategy(self, dice, target_category, rolls_remaining):
        """
        Optimized reroll strategy based on target category.
        Expects dice as list of ints in 0-5 (will be converted to 1-6).
        Returns a list of 5 binary values (1 = reroll, 0 = keep).
        """
        if rolls_remaining == 0:
            return [0, 0, 0, 0, 0]  # No rerolls left
        
        # Convert dice from 0-5 to 1-6
        dice = [d + 1 for d in dice]
        dice_counter = Counter(dice)
        
        category_info = self.category_scores[target_category]
        category_type = category_info['type']
        
        # Default strategy: reroll all dice
        reroll = [1, 1, 1, 1, 1]
        
        # Upper section (ones through sixes)
        if category_type == 'upper':
            target_value = category_info['number']
            
            # Keep all dice of target value
            for i, value in enumerate(dice):
                if value == target_value:
                    reroll[i] = 0

        # Three of a Kind
        elif category_type == 'three_kind':
            most_common = dice_counter.most_common(2)
            
            # Already have three or more of a kind
            if most_common and most_common[0][1] >= 3:
                value_to_keep = most_common[0][0]
                # Keep the three of a kind
                for i, value in enumerate(dice):
                    if value == value_to_keep:
                        reroll[i] = 0
                        
                # With remaining dice, keep high values if last roll
                if rolls_remaining == 1:
                    for i, value in enumerate(dice):
                        if value != value_to_keep and value >= 5:
                            reroll[i] = 0
            
            # Have a pair
            elif most_common and most_common[0][1] == 2:
                value_to_keep = most_common[0][0]
                
                # If multiple pairs, keep the higher pair
                if len(most_common) > 1 and most_common[1][1] == 2:
                    if most_common[0][0] < most_common[1][0]:
                        value_to_keep = most_common[1][0]
                
                # Keep the pair
                for i, value in enumerate(dice):
                    if value == value_to_keep:
                        reroll[i] = 0
                
                # If it's the last roll, also keep high values
                if rolls_remaining == 1:
                    for i, value in enumerate(dice):
                        if reroll[i] == 1 and value >= 5:
                            reroll[i] = 0
            
            # No pairs yet, but last roll - keep highest value
            elif rolls_remaining == 1:
                highest_value = max(dice) if dice else 6
                for i, value in enumerate(dice):
                    if value == highest_value:
                        reroll[i] = 0
                        break

        # Four of a Kind
        elif category_type == 'four_kind':
            most_common = dice_counter.most_common(1)
            
            # Already have four or more of a kind
            if most_common and most_common[0][1] >= 4:
                value_to_keep = most_common[0][0]
                # Keep the four of a kind
                kept = 0
                for i, value in enumerate(dice):
                    if value == value_to_keep and kept < 4:
                        reroll[i] = 0
                        kept += 1
                        
                # With remaining dice, keep high values if last roll
                if rolls_remaining == 1:
                    for i, value in enumerate(dice):
                        if reroll[i] == 1 and value >= 5:
                            reroll[i] = 0
            
            # Have three of a kind
            elif most_common and most_common[0][1] == 3:
                value_to_keep = most_common[0][0]
                # Keep the three of a kind
                for i, value in enumerate(dice):
                    if value == value_to_keep:
                        reroll[i] = 0
            
            # Have a pair and more rolls remaining
            elif most_common and most_common[0][1] == 2:
                # With multiple rolls, keep highest pair
                high_pair = 0
                for val, count in dice_counter.items():
                    if count == 2 and val > high_pair:
                        high_pair = val
                
                if high_pair > 0:
                    for i, value in enumerate(dice):
                        if value == high_pair:
                            reroll[i] = 0
            
            # Last roll and no good combos - keep highest value dice
            elif rolls_remaining == 1:
                sorted_dice = sorted(enumerate(dice), key=lambda x: x[1], reverse=True)
                for idx, _ in sorted_dice[:1]:  # Keep the highest die
                    reroll[idx] = 0

        # Full House
        elif category_type == 'full_house':
            # Already have a full house
            if len(dice_counter) == 2 and 2 in dice_counter.values() and 3 in dice_counter.values():
                reroll = [0, 0, 0, 0, 0]  # Keep all
            else:
                counts = dice_counter.most_common(2)
                
                # Have three of a kind and a different pair
                if len(counts) == 2 and counts[0][1] >= 3 and counts[1][1] >= 2:
                    reroll = [0, 0, 0, 0, 0]  # Keep all
                
                # Have three of a kind - keep it and try for a pair
                elif len(counts) >= 1 and counts[0][1] >= 3:
                    three_kind_value = counts[0][0]
                    
                    # Keep the three of a kind
                    for i, value in enumerate(dice):
                        if value == three_kind_value:
                            reroll[i] = 0
                    
                    # If we also have a single of a different value and last roll, keep it
                    if rolls_remaining == 1 and len(counts) > 1:
                        other_value = counts[1][0]
                        kept = 0
                        for i, value in enumerate(dice):
                            if value == other_value and reroll[i] == 1 and kept < 2:
                                reroll[i] = 0
                                kept += 1
                
                # Have two pairs - keep both pairs
                elif len(counts) >= 2 and counts[0][1] == 2 and counts[1][1] == 2:
                    for i, value in enumerate(dice):
                        if value == counts[0][0] or value == counts[1][0]:
                            reroll[i] = 0
                
                # Have one pair - keep it
                elif len(counts) >= 1 and counts[0][1] == 2:
                    pair_value = counts[0][0]
                    
                    # Keep the pair
                    for i, value in enumerate(dice):
                        if value == pair_value:
                            reroll[i] = 0
                    
                    # If last roll and we have a single of another value, keep it too
                    if rolls_remaining == 1 and len(counts) > 1:
                        other_values = [val for val, _ in counts[1:]]
                        highest_other = max(other_values)
                        kept = 0
                        for i, value in enumerate(dice):
                            if value == highest_other and reroll[i] == 1 and kept < 1:
                                reroll[i] = 0
                                kept += 1

        # Small Straight
        elif category_type == 'small_straight':
            values_set = set(dice)
            
            # Check how close we are to each large straight
            low_straight = [1, 2, 3, 4]
            mid_straight = [2, 3, 4, 5]
            high_straight = [3, 4, 5, 6]
            
            low_matches = []
            mid_matches = []
            high_matches = []
            for value in values_set:
                if value in low_straight:
                    low_matches.append(value)
                if value in high_straight:
                    high_matches.append(value)
                if value in mid_straight:
                    mid_matches.append(value)
                    
            low_matches = list(set(low_matches))
            high_matches = list(set(high_matches))
            mid_matches = list(set(mid_matches))
            
            maxm = max(len(low_matches), len(mid_matches), len(high_matches))
            maxm_list = []
            if len(low_matches) == maxm:
                maxm_list = low_matches
            elif len(mid_matches) == maxm:
                maxm_list = mid_matches
            elif len(high_matches) == maxm: 
                maxm_list = high_matches
                                
            for die in dice:
                if die in maxm_list:
                    reroll[dice.index(die)] = 0
                    maxm_list.remove(die)  # Remove to avoid duplicates

        # Large Straight
        elif category_type == 'large_straight':
            values_set = set(dice)
            
            # Already have a large straight
            if (all(v in values_set for v in [1, 2, 3, 4, 5]) or 
                all(v in values_set for v in [2, 3, 4, 5, 6])):
                reroll = [0, 0, 0, 0, 0]  # Keep all
            else:
                # Check how close we are to each large straight
                low_straight = [1, 2, 3, 4, 5]
                high_straight = [2, 3, 4, 5, 6]
                
                low_matches = []
                high_matches = []
                for value in values_set:
                    if value in low_straight:
                        low_matches.append(value)
                    if value in high_straight:
                        high_matches.append(value)
                low_matches = list(set(low_matches))
                high_matches = list(set(high_matches))
                
                if len(low_matches) >= len(high_matches):
                    for die in dice:
                        if die in low_matches:
                            reroll[dice.index(die)] = 0
                            low_matches.remove(die)  # Remove to avoid duplicates
                else:
                    for die in dice:
                        if die in high_matches:
                            reroll[dice.index(die)] = 0
                            high_matches.remove(die)  # Remove to avoid duplicates

        # Yahtzee
        elif category_type == 'yahtzee':
            
            most_common = dice_counter.most_common(1)
            for i, value in enumerate(dice):
                if most_common and most_common[0][0] == value:
                    reroll[i] = 0
        

        # Chance - keep high values
        elif category_type == 'chance':
            # Always keep 6s
            for i, value in enumerate(dice):
                if value == 6:
                    reroll[i] = 0
            
            # Keep 5s
            for i, value in enumerate(dice):
                if value == 5 and reroll[i] == 1:
                    reroll[i] = 0
            
            # On last roll, also keep 4s
            if rolls_remaining == 1:
                for i, value in enumerate(dice):
                    if value == 4 and reroll[i] == 1:
                        reroll[i] = 0
            
            # If we're keeping too few dice and it's the last roll, keep 3s too
            if rolls_remaining == 1 and sum(1 for r in reroll if r == 0) <= 2:
                for i, value in enumerate(dice):
                    if value == 3 and reroll[i] == 1:
                        reroll[i] = 0
                        
        return reroll

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU()

        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)

    def forward(self, x):
        identity = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out += identity  # Residual connection
        return self.activation(out)

class TargetIntuitionNet(nn.Module):
    def __init__(self, input_dim, output_dim=13):
        super(TargetIntuitionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.res_block = ResidualBlock(128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.res_block(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class YahtzeeAgent:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        
        obs_dim = env.observation_space.shape[0]
        self.enhanced_dim = obs_dim + 22
        
        self.target_intuition_net = TargetIntuitionNet(self.enhanced_dim, 13).to(self.device)
        self.reroller = HardCodedStrategy()
        
        # Optimizer for the target intuition network
        self.optimizer = optim.Adam(self.target_intuition_net.parameters(), lr=0.0005)
        
        # Hyperparameters
        self.gamma = 1.0  # Full credit for future rewards (no discount)
        self.epsilon = 0.5  # Higher starting epsilon for more exploration
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.batch_size = 256  # Larger batch size for more stable learning
        self.buffer = deque(maxlen=200000)  # Larger buffer for more diverse experiences
        
        # Category names for reference
        self.categories = [
            'ones', 'twos', 'threes', 'fours', 'fives', 'sixes',
            'three_of_a_kind', 'four_of_a_kind', 'full_house',
            'small_straight', 'large_straight', 'yahtzee', 'chance'
        ]
        
        # Category score descriptions for the reroll strategy
        self.category_scores = {
            0: {'type': 'upper', 'number': 1},  # Ones
            1: {'type': 'upper', 'number': 2},  # Twos
            2: {'type': 'upper', 'number': 3},  # Threes
            3: {'type': 'upper', 'number': 4},  # Fours
            4: {'type': 'upper', 'number': 5},  # Fives
            5: {'type': 'upper', 'number': 6},  # Sixes
            6: {'type': 'three_kind', 'score': 'sum'},  # Three of a Kind
            7: {'type': 'four_kind', 'score': 'sum'},   # Four of a Kind
            8: {'type': 'full_house', 'score': 25},     # Full House
            9: {'type': 'small_straight', 'score': 30},  # Small Straight
            10: {'type': 'large_straight', 'score': 40}, # Large Straight
            11: {'type': 'yahtzee', 'score': 50},        # Yahtzee
            12: {'type': 'chance', 'score': 'sum'}       # Chance
        }
        
        # Statistics tracking
        self.stats = {
            'upper_bonus_achieved': 0,
            'total_games': 0,
            'category_usage': {cat: 0 for cat in self.categories},
            'scores': [],
            'target_switches': 0,  # Track how often the target category changes
            'final_category_matches_target': 0  # Track if final category matches initial target
        }
        
        # Timing information
        self.times = {
            'select_action': [],
            'optimize_model': []
        }

    def enhance_observation(self, observation):
        """Add derived features to the observation to help the agent learn better"""
        dice = observation[:5] + 1  # Convert 0-5 to 1-6
        categories_available = observation[5:18]
        rolls_remaining = observation[18]
        
        # Dice value counts
        dice_counts = np.zeros(6)
        for i in range(5):
            if 1 <= dice[i] <= 6:
                dice_counts[int(dice[i])-1] += 1
        
        # Key statistics about dice
        has_three_kind = int(any(count >= 3 for count in dice_counts))
        has_four_kind = int(any(count >= 4 for count in dice_counts))
        has_yahtzee = int(any(count == 5 for count in dice_counts))
        has_pair = int(any(count >= 2 for count in dice_counts))
        
        # Upper section scoring potential
        upper_potentials = np.zeros(6)
        for i in range(6):
            if categories_available[i] == 1:  # Category is available
                upper_potentials[i] = (i+1) * dice_counts[i]
        
        # Upper section bonus tracking (need 63+ for bonus)
        upper_filled = sum(1 for i in range(6) if categories_available[i] == 0)
        remaining_turns = sum(categories_available)
        
        # Calculate straight potential
        unique_values = sum(1 for count in dice_counts if count > 0)
        small_straight_potential = 1.0 if unique_values >= 4 else (unique_values / 4.0)
        large_straight_potential = 1.0 if unique_values >= 5 else (unique_values / 5.0)
        
        # Full house potential
        has_three = any(count == 3 for count in dice_counts)
        has_two = any(count == 2 for count in dice_counts)
        full_house_potential = 1.0 if (has_three and has_two) else 0.5 if has_three or has_two else 0.0
        
        # Game progress (normalized)
        game_progress = (13 - remaining_turns) / 13.0
        
        # Upper section bonus situation
        upper_score = 0
        for i in range(6):
            if categories_available[i] == 0:  # Category already filled
                # Try to extract the score from the environment if available
                if hasattr(self.env, 'scorecard'):
                    upper_score += self.env.scorecard.get(self.categories[i], 0)
        
        upper_bonus_threshold = 63
        upper_bonus_progress = min(1.0, upper_score / upper_bonus_threshold)
        
        # Estimated potential to reach upper bonus
        remaining_upper_potential = 0
        for i in range(6):
            if categories_available[i] == 1:
                # Use average expected value for each category
                remaining_upper_potential += min((i+1) * 3, (i+1) * 5)  # Conservative estimate
        
        upper_bonus_potential = min(1.0, (upper_score + remaining_upper_potential) / upper_bonus_threshold)
        
        # Combine original observation with derived features
        enhanced = np.concatenate([
            observation,
            dice_counts,
            upper_potentials,
            [has_three_kind, has_four_kind, has_yahtzee, has_pair],
            [small_straight_potential, large_straight_potential, full_house_potential],
            [game_progress, upper_bonus_progress, upper_bonus_potential]
        ])
        
        return enhanced.astype(np.float32)

    def select_target_category(self, observation, is_eval=False):
        """Select a target category based on the current observation"""
        enhanced_obs = self.enhance_observation(observation)
        state = torch.from_numpy(enhanced_obs).float().to(self.device)
        
        # Get available categories
        categories_available = observation[5:18]
        legal_categories = [i for i in range(13) if categories_available[i] == 1]
        
        if not legal_categories:
            return None  # No legal categories left
        
        # Use epsilon-greedy for exploration during training
        if not is_eval and random.random() < self.epsilon:
            return random.choice(legal_categories)
        
        with torch.no_grad():
            # Get Q-values for all categories
            q_values = self.target_intuition_net(state.unsqueeze(0)).squeeze(0)
            
            # Mask unavailable categories with large negative values
            mask = torch.ones(13, device=self.device) * -1000000
            for i in legal_categories:
                mask[i] = 0
            masked_q = q_values + mask
            
            # Select category with highest Q-value
            return torch.argmax(masked_q).item()

    def calculate_reroll_strategy(self, dice, target_category, rolls_remaining):
        
        return self.reroller.calculate_reroll_strategy(dice, self.categories[target_category], rolls_remaining)
     
    def select_action(self, observation, is_eval=False, target_category=None):
        """
        Select an action based on the current observation
        Returns: (is_category_selection, action)
        """
        start = time.time()
        
        dice = observation[:5]
        categories_available = observation[5:18]
        rolls_remaining = observation[18]
        
        # If no target provided, compute one
        if target_category is None:
            target_category = self.select_target_category(observation, is_eval)
        
        # When no rerolls are left or no categories are available for targeting,
        # we must select a category to play
        if rolls_remaining == 0 or all(categories_available[i] == 0 for i in range(13)):
            is_category_selection = True
            action = self.select_target_category(observation, is_eval)
        else:
            # Otherwise, calculate reroll pattern
            is_category_selection = False
            decisions = self.calculate_reroll_strategy(dice, target_category, rolls_remaining)
            action = int(''.join(map(str, decisions)), 2) + 13
            # action = reroll_pattern 
            # print("selected reroll action:", action)
            
        # Verify action is legal
        legal_actions = self.env.get_legal_actions()
        if action >= len(legal_actions) or legal_actions[action] != 1:
            # Fallback to a legal action
            legal_indices = [i for i, is_legal in enumerate(legal_actions) if is_legal == 1]
            if legal_indices:
                if action < 13:  # Category selection
                    is_category_selection = True
                    action = random.choice([i for i in legal_indices if i < 13])
                else:  # Reroll action
                    is_category_selection = False
                    # print("RANDOM REROLL REQUESTED!!")
                    action = random.choice([i for i in legal_indices if i >= 13])
        
        end = time.time()
        self.times['select_action'].append(end - start)
        
        # print("returning action:", action, target_category)
        return is_category_selection, action, target_category

    def shape_reward(self, reward, action, observation, next_observation, done, info):
        """Apply reward shaping to encourage better strategic play"""
        shaped_reward = reward
        dice = observation[:5] + 1  # Convert 0-5 to 1-6
        categories_available = observation[5:18]
        
        # Category selection (actions 0-12)
        if action < 13:
            # Track the chosen category
            category = self.categories[action]
            
            # Strategic zeros - less penalty if few options remain
            if reward == 0:
                remaining_categories = sum(categories_available)
                if remaining_categories <= 3:
                    shaped_reward = -2  # Less penalty for strategic zeros late game
                else:
                    shaped_reward = -5  # Standard penalty
            
            # Upper section scoring
            if action < 6:
                # Calculate current upper section total
                upper_total = 0
                if hasattr(self.env, 'scorecard'):
                    for i in range(6):
                        if i != action and not categories_available[i]:  # Category already filled
                            upper_total += self.env.scorecard.get(self.categories[i], 0)
                
                # Add current category score
                upper_with_current = upper_total + reward
                
                # Bonus for moves that help achieve upper bonus (threshold is 63)
                if upper_with_current >= 63:
                    shaped_reward += 40  # Strong bonus for achieving the upper bonus
                elif upper_with_current >= 55:
                    shaped_reward += 25  # Good progress toward bonus
                elif reward >= (action + 1) * 3:
                    shaped_reward += 15  # Reward good upper section scores
            
            # Reward efficient use of categories
            if category == 'yahtzee' and reward >= 50:
                shaped_reward += 30  # Extra bonus for Yahtzee
            elif category in ['small_straight', 'large_straight'] and reward > 0:
                shaped_reward += 15  # Bonus for straights
            elif category == 'full_house' and reward > 0:
                shaped_reward += 10  # Bonus for full house
            elif category in ['three_of_a_kind', 'four_of_a_kind'] and reward >= (3 if category == 'three_of_a_kind' else 4) * 4:
                shaped_reward += 5  # Bonus for good three/four of a kind
        
        # Terminal reward based on final score
        if done:
            total_score = info.get('total_score', 0)
            
            # Graduated rewards based on score thresholds
            if total_score >= 250:
                shaped_reward += 150  # Exceptional score
            elif total_score >= 200:
                shaped_reward += 80   # Excellent score
            elif total_score >= 150:
                shaped_reward += 40   # Good score
            elif total_score >= 120:
                shaped_reward += 20   # Decent score
                
            # Special bonus for achieving upper section bonus
            if hasattr(self.env, 'scorecard') and self.env.scorecard.get('upper_bonus', 0) > 0:
                shaped_reward += 50  # Strong encouragement for upper bonus
        
        return shaped_reward

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def evaluate(self, num_episodes=10):
        """Evaluate the agent's performance without exploration"""
        total_scores = []
        upper_bonus_count = 0
        category_usage = {cat: 0 for cat in self.categories}
        
        for _ in range(num_episodes):
            observation, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            # Initial target category selection
            target_category = self.select_target_category(observation, is_eval=True)
            
            while not done:
                # Select action based on target category
                is_category_selection, action, target_category = self.select_action(
                    observation, is_eval=True, target_category=target_category
                )
                
                # Track category usage
                if is_category_selection:
                    category_usage[self.categories[action]] += 1
                
                # Take action
                observation, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                # Recalculate target after each step if there are rerolls left
                if not done and observation[18] > 0:  # rolls_remaining > 0
                    target_category = self.select_target_category(observation, is_eval=True)
            
            total_scores.append(episode_reward)
            
            # Check if upper bonus was achieved
            if hasattr(self.env, 'scorecard') and 'upper_bonus' in self.env.scorecard:
                if self.env.scorecard['upper_bonus'] > 0:
                    upper_bonus_count += 1
        
        # Update statistics
        self.stats['scores'].extend(total_scores)
        self.stats['upper_bonus_achieved'] += upper_bonus_count
        self.stats['total_games'] += num_episodes
        
        # Merge category usage
        for cat, count in category_usage.items():
            self.stats['category_usage'][cat] += count
            
        return np.mean(total_scores)
    
class YahtzeeEvaluator:
    def __init__(self, agent, num_games=10):
        
        self.agent = agent
        self.env = agent.env
        self.num_games = num_games
        
        # Results storage
        self.game_logs = []
        self.category_stats = defaultdict(list)
        self.reroll_stats = []
        self.score_distribution = []
        
    def evaluate(self):
        """Evaluate the agent with detailed logging"""
        print(f"Running detailed evaluation over {self.num_games} games...")
        
        for game_id in range(self.num_games):
            game_log = self._play_single_game(game_id)
            self.game_logs.append(game_log)
            self._process_game_statistics(game_log)
            self._print_game_summary(game_log)
            print("-" * 80)
            
        self._print_overall_statistics()
        
        return {
            'game_logs': self.game_logs,
            'category_stats': dict(self.category_stats),
            'reroll_stats': self.reroll_stats,
            'score_distribution': self.score_distribution
        }
    
    def _play_single_game(self, game_id):
        """Play a single game with detailed tracking"""
        observation, _ = self.env.reset()
        done = False
        total_reward = 0
        step = 0
        
        # Game log structure
        game_log = {
            'game_id': game_id,
            'steps': [],
            'score': 0,
            'scorecard': {},
            'upper_bonus_achieved': False,
            'category_selections': []
        }
        
        # Target tracking
        target_category = self.agent.select_target_category(observation, is_eval=True)
        target_category_name = self.agent.categories[target_category]
        
        while not done:
            # Log current state
            step_log = {
                'step': step,
                'dice_before': observation[:5].astype(int) + 1,
                'target_category': target_category_name,
                'rolls_remaining': int(observation[18])
            }
            
            # Select action
            is_category_selection, action, target_category = self.agent.select_action(
                observation, is_eval=True, target_category=target_category
            )
            
            # Take action
            old_observation = observation.copy()
            
            oldobstemp = observation
            
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            done = terminated or truncated
            total_reward += reward
            
            # Log action results
            if is_category_selection:
                # Category selection
                category_idx = action
                category_name = self.agent.categories[category_idx]
                step_log['action_type'] = 'category_selection'
                step_log['category_selected'] = category_name
                step_log['score'] = reward
                
                # Track category selections
                game_log['category_selections'].append({
                    'category': category_name,
                    'score': reward,
                    'target_matched': category_name == target_category_name,
                    'dice': step_log['dice_before'].tolist()
                })
                
            else:
                # Reroll action
                reroll_pattern = action - 13
                # reroll_indexes = [i for i in range(5) if reroll_pattern & (1 << i)]
                reroll_indexes =  [int(bit) for bit in format(reroll_pattern, '05b')]
                reroll_indexes = [i for i in range(5) if reroll_indexes[i] == 1]
                
                # print("ABHAY Rerolled indexes:", reroll_indexes)
                
                step_log['action_type'] = 'reroll'
                step_log['reroll_indexes'] = reroll_indexes
                step_log['dice_after'] = observation[:5].astype(int) + 1
                
                # Track reroll effectiveness
                for i in reroll_indexes:
                    self.reroll_stats.append({
                        'game_id': game_id,
                        'step': step,
                        'die_position': i,
                        'value_before': old_observation[i] + 1,
                        'value_after': observation[i] + 1,
                        'target_category': target_category_name
                    })
            
            # Recalculate target if there are rerolls left
            if not done and observation[18] > 0:
                new_target = self.agent.select_target_category(observation, is_eval=True)
                new_target_name = self.agent.categories[new_target]
                
                step_log['target_changed'] = new_target_name != target_category_name
                step_log['new_target'] = new_target_name if step_log.get('target_changed', False) else None
                
                target_category = new_target
                target_category_name = new_target_name
            
            # Add step log to game log
            game_log['steps'].append(step_log)
            step += 1
        
        # Final game statistics
        game_log['score'] = total_reward
        
        # Extract scorecard if available
        if hasattr(self.env, 'scorecard'):
            game_log['scorecard'] = dict(self.env.scorecard)
            game_log['upper_bonus_achieved'] = self.env.scorecard.get('upper_bonus', 0) > 0
        
        self.score_distribution.append(total_reward)
        
        return game_log
    
    def _process_game_statistics(self, game_log):
        """Extract and process statistics from a game log"""
        # Process category selections
        for selection in game_log['category_selections']:
            category = selection['category']
            self.category_stats[category].append({
                'score': selection['score'],
                'target_matched': selection['target_matched'],
                'dice': selection['dice'],
                'game_id': game_log['game_id']
            })
    
    def _print_game_summary(self, game_log):
        """Print a human-readable summary of a game"""
        print(f"\nGame {game_log['game_id']+1} Summary - Final Score: {game_log['score']}")
        
        # Print each step
        steps_data = []
        for step in game_log['steps']:
            if step['action_type'] == 'category_selection':
                steps_data.append([
                    step['step'],
                    'Category',
                    f"{' '.join(map(str, step['dice_before']))}",
                    f"Selected {step['category_selected']} (Target: {step['target_category']})",
                    f"{step['score']} points"
                ])
            else:  # reroll
                reroll_str = ', '.join([str(i+1) for i in step['reroll_indexes']])
                dice_before = ' '.join(map(str, step['dice_before']))
                dice_after = ' '.join(map(str, step['dice_after']))
                
                target_info = step['target_category']
                if step.get('target_changed', False):
                    target_info += f" → {step['new_target']}"
                
                steps_data.append([
                    step['step'],
                    'Reroll',
                    f"{dice_before} → {dice_after}",
                    f"Rerolled positions: {reroll_str}",
                    f"Target: {target_info}"
                ])
        
        print(tabulate(steps_data, headers=['Step', 'Action', 'Dice', 'Details', 'Result']))
    
    def _print_overall_statistics(self):
        """Print overall statistics from the evaluation"""
        scores = self.score_distribution
        
        print("\n======== EVALUATION SUMMARY ========")
        print(f"Games played: {self.num_games}")
        print(f"Average score: {np.mean(scores):.1f} (min: {min(scores)}, max: {max(scores)})")
        print(f"Median score: {np.median(scores):.1f}")
        print(f"Standard deviation: {np.std(scores):.1f}")
        

def run_yahtzee_evaluation(env, device, model_path="best_yahtzee_target_model.pt", num_games=10):

    agent = YahtzeeAgent(env, device)
    
    # Load pretrained model
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    agent.target_intuition_net.load_state_dict(checkpoint["target_intuition_net"])
    
    # Create and run evaluator
    evaluator = YahtzeeEvaluator(agent, num_games=num_games)
    evaluator.evaluate()
    
    return evaluator

if __name__ == "__main__":
    num_games = 1000
    print(f"Running evaluation script for {num_games} games...")
    
    env = YahtzeeEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r"PretrainedIntuitionNet.pt"
    
    evaluator = run_yahtzee_evaluation(env, device, model_path=model_path, num_games=num_games)
