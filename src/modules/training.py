import numpy as np
from ..app.qlearning_agent import discretize_state


def training(env, q_agent, buckets, episodes, debug=False):
    if debug:
        log_file = open("log/training.log", "w")

    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretize_state(env, buckets, obs)
        step = 0
        done = False
        while not done:
            action = q_agent.choose_action(state)
            
            if debug:
                # Monitor ε-greedy choice
                if np.random.uniform(0, 1) < q_agent.epsilon:
                    log_message = f"Episode {episode+1}, Step {step}: Exploring with action {action}\n"
                else:
                    log_message = f"Episode {episode+1}, Step {step}: Exploiting with action {action}\n"
                
                log_file.write(log_message)

            next_obs, reward, done, _, _ = env.step(action)
            next_state = discretize_state(env, buckets, next_obs)
            
            if debug:
                # Monitor Q-Table Updates
                old_value = q_agent.q_table[*state, action]
                q_agent.learn(state, action, reward, next_state)
                new_value = q_agent.q_table[*state, action]
                log_message = f"Episode {episode+1}, Step {step}: State: {state}, Action: {action}, Reward: {reward}, Old Q-value: {old_value}, New Q-value: {new_value}\n"
                log_file.write(log_message)
            else:
                q_agent.learn(state, action, reward, next_state)
            
            state = next_state
            step += 1

        if (episode + 1) % 100 == 0:
            log_message = f"Episode {episode+1} completed\n"
            print(log_message, end='')
            if debug:
                log_file.write(log_message)

    if debug:
        log_file.close()
