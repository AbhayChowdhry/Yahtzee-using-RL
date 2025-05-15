import gradio as gr
import numpy as np
import torch
from YahtzeeEnv import YahtzeeEnv  
from evaluator import YahtzeeAgent  

env = YahtzeeEnv()
obs, _ = env.reset()
path = r"PretrainedIntuitionNet.pt"
agent = YahtzeeAgent(env, device="cpu")
checkpoint = torch.load(path, map_location=torch.device("cpu"))
agent.target_intuition_net.load_state_dict(checkpoint["target_intuition_net"])

def update_ui_from_env():
    obs = env._get_observation()
    dice = (obs[:5] + 1).tolist()
    categories = obs[5:18].tolist()
    remaining_rolls = int(obs[18])

    # Prevent crash if game is over and obs is not usable
    try:
        target_value = int(agent.select_target_category(obs, is_eval=True))
    except:
        target_value = -1  # Indicate no target available

    try:
        obs_enhanced = agent.enhance_observation(obs)
        obs_tensor = torch.tensor(obs_enhanced, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.target_intuition_net(obs_tensor).cpu().numpy().squeeze()
        q_values_categories = q_values[:13].tolist()
        max_idx = np.argmax(q_values_categories)
        q_values_md = [
            f"<span style='background-color: yellow; color: black; font-weight: bold'>{q:.2f}</span>" if i == max_idx else f"{q:.2f}"
            for i, q in enumerate(q_values_categories)
        ]
    except:
        q_values_md = ["N/A"] * 13

    return tuple(dice + categories + [remaining_rolls, target_value] + q_values_md)


def update_env_from_ui(
    die1, die2, die3, die4, die5,
    cat1, cat2, cat3, cat4, cat5, cat6, cat7,
    cat8, cat9, cat10, cat11, cat12, cat13,
    remaining_rolls
):
    dice_vals = [die1-1, die2-1, die3-1, die4-1, die5-1]
    category_vals = [
        int(cat1), int(cat2), int(cat3), int(cat4), int(cat5),
        int(cat6), int(cat7), int(cat8), int(cat9), int(cat10),
        int(cat11), int(cat12), int(cat13)
    ]
    env.dice = np.array(dice_vals, dtype=np.int8)
    env.available_categories = np.array(category_vals, dtype=np.int8)
    env.remaining_rolls = int(remaining_rolls)
    return update_ui_from_env() + (0, False, "")  # (reward, game_over, last_action)

def take_action():
    obs = env._get_observation()
    target, action, target_category = agent.select_action(obs, is_eval=True)
    # Step the environment using the chosen action.
    obs, reward, done, _, info = env.step(action)
    
    if done:
        print("Restarting new game...")
        reset_game()
    
    action_display = ""
    if action < 13:
        action_display = f"{list(env.categories.keys())[action]}"
    else:
        reroll_indexes =  [int(bit) for bit in format(action-12, '05b')]
        reroll_indexes = [i+1 for i in range(5) if reroll_indexes[i] == 1]
        action_display = f"Rerolling dices {reroll_indexes}"
    
    # Create a description of the last action taken.
    # last_action = f"Last performed action: {action_display}"
    last_action = f"<div style='font-size: 30px; font-weight: bold;'>Last performed action: {action_display}</div>"
    
    updated_ui = update_ui_from_env()
    # Return updated UI plus the reward, game over flag, and last action message.
    return updated_ui + (reward, done, last_action)

def reset_game():
    env.reset()
    updated_ui = update_ui_from_env()
    # On reset, reward is 0, game is not over, and last action message is cleared.
    return updated_ui + (0, False, "")

# Build UI
with gr.Blocks(css=".gradio-container { max-width: 1500px; margin: auto; }") as demo:
    gr.Markdown("# 🎲 Yahtzee RL Environment")
    gr.Markdown("""
    ### 1. **Manual Update:** To start click Restart Game. Adjust the dice values, category checkboxes, and remaining rolls, then click **Apply Manual Changes**. This also updates the target value and expected Q-values.
    ### 2. **Execute Action:** Click **Execute Action** to update the game state, which in turn refreshes the Q-values and displays the last action taken.
    ### 3. **Reset Game:** Click **Reset Game** to start a new game.
    ### 4. **Game Over:** If the game is over, the UI will AUTOMATICALLY reset, and you can start a new game.
    ### 5. Kindly DO NOT input wrong entries since the game might crash, in which case restart from the terminal
    ### (The reason we do not score total score or history because it's meaningless if we're manually adjusting the game state)
    """)

    # Add custom CSS for much larger action text
    gr.HTML("<style>.extra-large-text { font-size: 1.8em; font-weight: bold; color: #2a4d69; }</style>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Dice Values")
            # Editable number fields for dice values
            dice_inputs = [gr.Number(label=f"Die {i+1}", value=0) for i in range(5)]
        with gr.Column():
            gr.Markdown("### Categories: Availability + Q-Values")
            combined_category_outputs = []
            category_checks = []
            qvalue_outputs = []
            for cat in env.categories.keys():
                with gr.Row():
                    check = gr.Checkbox(value=True, label=cat.capitalize())
                    qval = gr.Markdown(value="0.00")
                    category_checks.append(check)
                    qvalue_outputs.append(qval)
                    combined_category_outputs.append((check, qval))

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Remaining Rolls")
            remaining_rolls_input = gr.Number(label="Remaining Rolls", value=2)
        with gr.Column():
            gr.Markdown("### Target Category")
            target_value_output = gr.Number(label="Target Index", interactive=False)

    with gr.Row():
        gr.Markdown("#### Manual Update")
        manual_update = gr.Button("🛠️ Apply Manual Changes")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Action")
            step_button = gr.Button("🚀 Execute Action")
        with gr.Column():
            gr.Markdown("### Reset Game")
            reset_button = gr.Button("🔄 Reset Game")
    
    # Create shared reward, game_over, and last_action components
    with gr.Row():
        with gr.Column():
            reward_display = gr.Number(label="Reward", value=0)
        with gr.Column():
            game_over_display = gr.Checkbox(label="Game Over", value=False)
    
    with gr.Row():
        last_action_display = gr.Markdown(label="Last Action", value="")
    
    # Define button actions
    manual_update.click(
        fn=update_env_from_ui,
        inputs=[
            *dice_inputs,          # 5 dice values
            *category_checks,      # 13 category availabilities
            remaining_rolls_input  # remaining rolls
        ],
        outputs=(
            dice_inputs +
            category_checks +
            [remaining_rolls_input, target_value_output] +
            qvalue_outputs +
            [reward_display, game_over_display, last_action_display]
        )
    )
    
    step_button.click(
        fn=take_action,
        outputs=(
            dice_inputs +
            category_checks +
            [remaining_rolls_input, target_value_output] +
            qvalue_outputs +
            [reward_display, game_over_display, last_action_display]
        )
    )
    
    reset_button.click(
        fn=reset_game,
        outputs=(
            dice_inputs +
            category_checks +
            [remaining_rolls_input, target_value_output] +
            qvalue_outputs +
            [reward_display, game_over_display, last_action_display]
        )
    )
    
demo.launch()