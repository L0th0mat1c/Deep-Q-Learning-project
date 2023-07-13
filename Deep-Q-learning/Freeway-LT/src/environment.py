from email import utils
import time
import gymnasium as gym
import src.agents as agents

def get_env():
    env = gym.make('Freeway-ram-v4', render_mode="rgb_array")
    state = env.reset()
    
    return (env, state)


def run(Agent: agents.Agent, render: bool=False, n_runs: int=1, verbose=True):
    scores = []  # List of each run rewards
    for i in range(n_runs):
        env, initial_state = get_env()
        agent = Agent()

        game_over = False
        next_action = agent.act(initial_state, 0)

        while not game_over:
            if render:
                time.sleep(0.025)
                env.render()

            ob, reward, game_over, _ = env.step(next_action)
            next_action = agent.act(ob, reward)


        # Small hack: The byte 103 contains the Player 1 score.
        player_score = utils.convert_score(ob[103])

        if verbose:
            print(f"Score #{i}: {player_score}")

        scores.append(player_score)

        env.close()
        
    return scores