## Using Reinforcement learning to play the game of Yahtzee 

### A Wild Journey into Sparse Rewards, Broken Bots, and Dicey Decisions

So, Yahtzee. That classic game of dice, decisions, and the occasional desire to flip the table when you *just can't* roll that blasted Yahtzee. 

If you arent familiar with the game, you can check it out at [Wikipedia](https://en.wikipedia.org/wiki/Yahtzee) or [here](https://cardgames.io/yahtzee/how-to-play-yahtzee/). Here is a summary:

The objective of the game is to score points by rolling five dice to make certain combinations. The dice can be rolled up to three times in a turn to try to make various scoring combinations and dice must remain in the box. A game consists of thirteen rounds. After each round, the player chooses which scoring category is to be used for that round. Once a category has been used in the game, it cannot be used again. The scoring categories have varying point values, some of which are fixed values and others for which the score depends on the value of the dice. A Yahtzee is five-of-a-kind and scores 50 points, the highest of any category. The winner is the player who scores the most points.

Here is an example of the game:

![alt text](https://github.com/AbhayChowdhry/Yahtzee-using-RL/blob/main/Images/GameExample.jpg)
Image taken from https://cardgames.io/yahtzee/

I decided to take on the challenge of teaching an AI to master it, not with brute-force calculations, but with the "learn-by-doing" magic of Reinforcement Learning. (We maximize the score of a single agent and not a versus scenario as shown)

### First Off, What's Reinforcement Learning (RL) and Why Yahtzee?

In a nutshell, RL is like training a dog. The AI (our "agent") tries to perform a task in an environment (playing Yahtzee). When it does something good (like scoring a high Full House), it gets a "treat" (a positive reward). When it does something... less than ideal (like scoring a 0 in Yahtzee when it had four-of-a-kind), it might get a "scolding" (a negative reward, or just no treat). Over many, *many* attempts, the agent learns a "policy" – a strategy to maximize its cumulative treats.

Check out Sutton and Barto's book [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) for a deep dive into the theory. It's a classic!

Why Yahtzee? Well, it's a fascinating beast!
1.  **Long-term planning is key:** Do you sacrifice a mediocre score in "Threes" now to keep your options open for a Full House later? These kinds of decisions have consequences that ripple through the game.
2.  **Sparse Rewards:** This is a big one. In Yahtzee, you roll, you re-roll, you re-roll again... and you only get points *after* you decide which category to fill. Most actions (re-rolling dice) don't give you immediate feedback on whether they were "good" for your final score. A game has 13 turns, and in each turn, you might make two re-roll decisions before picking a category. That means roughly two-thirds of your key decisions per turn yield an immediate reward of $0$! The agent has to connect these "silent" actions to eventual glory (or doom). The total state space isn't astronomical compared to [Go](https://deepmind.google/research/breakthroughs/alphago/) or [Chess](https://en.wikipedia.org/wiki/AlphaZero), but it's chunky enough to be interesting.

For all my algorithm friends out there: Yes! We can solve Yahtzee using traditional algorithmic approaches-such as exhaustive search and dynamic programming, which maximize expected scores- however the idea of using RL is to sow the seeds for these models to tackle simpler problems such as yahtzee (which have model based equivalents) so that one day we can reap the fruit of solving potentially world changing problems such mental health, mathematics, analytical reasoning (which do not have algorithms!). Check out [Model Free RL](https://en.wikipedia.org/wiki/Model-free_(reinforcement_learning))

Now, state-of-the-art methods like [Monte Carlo Tree Search (MCTS)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) could probably crack Yahtzee and hit those sweet ~250 average scores. But where's the fun in that? **My challenge to myself: conquer Yahtzee using "just" Deep Q-Networks (DQN) and its clever modifications. Spoiler: It's harder than it sounds.**

### Building the Playground (The Yahtzee Environment)

Every RL agent needs an environment to play in. So, the first order of business was coding up a digital Yahtzee game. Dice, score sheet, rules – the whole shebang.

Observation space:
* The current dice (5 values)
* Number of re-rolls left (1 value)
* Which categories on the scorecard are already filled (13 binary values – 1 if filled, 0 if open).

Action space:
* Choose one of the 13 categories to score.
* Choose which dice to re-roll (there are $2^5 = 32$ combinations, from re-rolling none to re-rolling all).

The initial version allowed illegal actions, and provided a -50 penalty for them. We talk about this later.

### Attempt #1: The "Flatten Everything and Pray" Basic DQN

My first foray was a straightforward Deep Q-Network. 

Deep Q-Networks are a type of neural network that approximates the Q-value function (Q value is the expected future reward of taking a certain action in a certain state). The DQN takes the current state (observation) and outputs Q-values for all possible actions. The agent then picks the action with the highest Q-value.

This "observations" and "actions" was flattened into a simple list of numbers. 

The DQN would take the flattened observation and try to predict the best action out of these 45 (13 + 32).

![alt text](https://github.com/AbhayChowdhry/Yahtzee-using-RL/blob/main/Images/DQN_1.jpg)
Example illustration of the DQN architecture. 

DQN use a lot of interesting tricks to stabilize training, like experience replay and target networks. I won't go into all the details here, but if you're interested, check out the [original DQN paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) by Mnih et al.

This is where I first realsed that RL is a **hyperparameter tuning drama queen**! "Is my model architecture flawed, or are my learning rate / batch size / epsilon decay just wrong?" became my morning mantra. After much tweaking (and I mean *much*), this initial approach clawed its way from a measly average score of about 45 to a slightly-less-measly 70. One surprising MVP throughout this whole journey? The learning rate. That thing is sensitive!

Needless to say, this was a *very* naive approach. The agent had no clue what it was doing. It was like trying to teach a toddler to play chess by showing them a board and saying, "Good luck!"

> A crucial aside: **"I DO NOT WANT TO ENCODE MY STRATEGIC BIASES!"** If I start rewarding the agent for, say, getting more than the "average" score for a category, that's *my* strategy, not its learned wisdom. If I wanted to encode strategies, I'd just write a dynamic programming solver and call it a day! The urge to inject my own biases was strong, a constant temptation to resist.

### Attempt #2: "Okay, Let's Think Like a Human (Sort Of)" - The HRL Diversion

Scoring 70 is... well, it's not going to win you any Yahtzee tournaments. I started thinking about how *I* play. 

One major insight from this introspection: the act of re-rolling dice and the act of choosing a category to score feel *very* different. Just flattening them into one list of 45 actions felt... crude. Maybe the agent was struggling to distinguish between these fundamentally different decision types.

Enter **Hierarchical Reinforcement Learning (HRL)**.
> **HRL in a Nutshell:** Imagine a company. There's a CEO (high-level policy) who decides on broad goals ("Let's increase market share in Q3!"). Then there are managers (low-level policies) who figure out *how* to achieve those goals ("Launch a new marketing campaign," "Develop product X"). HRL is similar: a top-level agent picks sub-goals, and lower-level agents learn to achieve them.

I stumbled upon a [Stanford paper](https://web.stanford.edu/class/aa228/reports/2018/final75.pdf) where they used HRL for Yahtzee and got a score around 120-130. Promising! So, I built an HRL setup:
1.  A "manager" network that decided: "Should I re-roll, or should I pick a category to score?"
2.  If "re-roll": A "specialist" network decided *which* of the 32 dice combinations to re-roll.
3.  If "pick category": Another "specialist" network decided *which* of the 13 categories to fill.

![alt text](https://github.com/AbhayChowdhry/Yahtzee-using-RL/blob/main/Images/HRL.jpg)

I even tried simplifying the re-roll specialist's job by having it output 5 values (one for each die: re-roll or keep) instead of 32 combinations.
The result? Still stuck at 70. "This was a bummer," is an understatement. 😖

### The HRL Post-Mortem: A Flawed Manager

Digging into the HRL agent's behavior, I noticed it was still making a *lot* of illegal moves during evaluation (like trying to pick an already filled category). The issue seemed to lie with the "manager" network. I hadn't really solved the long-term planning problem; I'd just pushed it onto another network!
As a quick fix, I started masking out illegal actions (instead of just giving a -50 penalty, which sometimes led to the agent getting stuck in infinite loops of negative rewards, bless its digital heart). This helped stability but didn't improve the score.

### Attempt #3: Learning Like a Baby - Curriculum Learning

It was clear the agent wasn't grasping long-term patterns. It was playing a locally optimal game, grabbing whatever points it could *right now*, which often led to filling many categories with 0s towards the end.

Enter **Curriculum Learning**!
> **Curriculum Learning Explained:** Think about how humans learn. We don't start with calculus; we start with 1+1=2. Curriculum Learning applies this to AI. You start training the agent on a very simple version of the problem (the "curriculum") and gradually increase the complexity. This is especially handy for sparse reward scenarios like ours, as it can guide the agent towards understanding basic reward structures before tackling the full, daunting task.

I designed a curriculum – perhaps starting with simpler goals or fewer turns. Here is an example curriculum I used:

| Curriculum Stage | Description                                   | Episodes     |
|------------------|-----------------------------------------------|--------------|
| Stage 1          | Upper section only             | 0-2500       |
| Stage 2          | Three of a kind, four of a kind, full house   | 2500-5000    |
| Stage 3          | Small / Large straight                        | 5000-7500    |
| Stage 4          | All                                            | 7500-20000   |

A lot of  changes were tested with varying sizes of episodes, modifying learning rates as stages progress etc. Upon incorporating this, *finally*, a breakthrough! The score jumped to around 100-110. We were learning!

### Attempt #4: The "Dynamic Intuition" Epiphany

To get further, I did something... surprisingly fun. I played Yahtzee. A lot. And I meticulously mapped out my *exact* thought process for each action. This led to a critical insight:
>**“Whenever I re-roll a dice, I re-roll based on a category I'm targeting. Based on the obtained dice, I either select that category or change my target!”**

This sparked an idea for what I call **"Dynamic Intuition,"** a custom modification to the DQN framework.
The core idea: The agent has an "intuition" (a target category) about what to aim for. As the environment (dice rolls) changes, its intuition adapts! Much like life, really. You aim for one thing, life throws you a curveball, and you adjust your plans with the new information.

![alt text](https://github.com/AbhayChowdhry/Yahtzee-using-RL/blob/main/Images/DynamicIntuition.jpg)

From the HRL struggles, I learned that a classifier deciding *between* re-rolling and choosing a category was problematic. This is when another lightbulb went on: **“Given a target category and the current dice, deciding what to re-roll is almost part of the game's rules or a very constrained problem!”** (Sure, you could train separate RL agents for each target category to decide re-rolls, but that felt like overkill given how given a category, the reroll stratergy is extremely straight forward).

This led to a crucial design choice that sidestepped the need for an HRL-style manager:
* The agent *always* has a target category in mind.
* It *always* goes through (up to) two re-roll phases.
* Even if the target is perfectly achieved after the first roll (e.g., Yahtzee!), for the sake of a consistent process, it "re-rolls" zero dice. This eliminates the need for a network to decide *if* to re-roll.

Here's the "Dynamic Intuition" model flow:
1.  An **Intuition Network (DQN)** observes the board (filled categories) and current dice (if any, from previous turn or initial roll) and selects a **target category**.
2.  Based on this target and the current dice, a **hard-coded (or very simple, pre-defined) re-roll logic** determines which dice to re-roll. (e.g., if targeting Yahtzee and you have three 6s, keep the 6s, re-roll the others). It executes the first re-roll.
3.  The Intuition Network looks at the new dice and the board again, and selects an **updated target category**.
4.  The hard-coded re-roll logic executes the second re-roll based on this new target.
5.  Finally, the Intuition Network looks at the dice one last time and selects the category to score. This category is then played. Note: the *final* decision is to pick a category, not whether to reroll.

The first few implementations, along with some much-needed feature enhancements (moving beyond simple flattened observations to more structured input features), rocketed the score to about 140-150! This jump was exhilarating.

### Attempt #5: The Power Combo - Dynamic Intuition + Curriculum Learning

![alt text](https://github.com/AbhayChowdhry/Yahtzee-using-RL/blob/main/Images/FinalArch.jpg)

We had two successful strategies: Dynamic Intuition gave us a better decision-making framework, and Curriculum Learning helped the agent learn fundamentals. What if we combined them?
The hypothesis: Curriculum Learning could help the Dynamic Intuition model develop its "intuition" more effectively, starting with simpler target-seeking behaviors before tackling the full game's complexity.
It worked! With this combo and further painstaking hyperparameter tuning, we pushed the average score to **180!**

### The Final Push: Network Size, Patience, and a Dash of Luck

We were getting close to "respectable" Yahtzee scores. I meticulously examined all components. I experimented with different network sizes for the Intuition Net and varied episode lengths for training (initially 10-30k episodes, which felt like a lot at the time).
The final breakthrough came from training a single ResNet (Residual Network, good for deeper networks) for the Intuition Net for a significantly longer period – around **100,000 episodes**.
This pushed our agent to its peak performance: an average score of **205!**

Interestingly, bigger networks sometimes converged to lower optimal scores (even with smaller learning rates – go figure!), and smaller networks just couldn't seem to break past the 190 barrier. It seems there's a "Goldilocks zone" for network size in this problem.

I built a little interactive page where you can vizualize the live Q-values for each step.

![alt text](https://github.com/AbhayChowdhry/Yahtzee-using-RL/blob/main/Images/Demo.jpg)

Check it out [here](https://huggingface.co/spaces/abhaych/YahtzeeRL). Ping me if the link doesnt work (it sleeps after inactivity for a while)
