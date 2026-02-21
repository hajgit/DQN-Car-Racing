# train_1600ep_fast.py
import os
import torch
import datetime
import gymnasium as gym
import gymnasium.wrappers as gym_wrap
import numpy as np
from DQN_model import Agent, SkipFrame

# Optimisations système
os.environ["SDL_VIDEODRIVER"] = "dummy"
torch.set_num_threads(1)

def main():
    print("⚡ Démarrage de l'entraînement OPTIMISÉ (1600 épisodes)...")
    
    # Environnement optimisé
    env = gym.make(
        "CarRacing-v3",
        continuous=False,
        render_mode=None,
        max_episode_steps=800  # Évite les épisodes trop longs
    )
    env = SkipFrame(env, skip=5)  # ⚡ Plus rapide que skip=4
    env = gym_wrap.GrayscaleObservation(env)
    env = gym_wrap.ResizeObservation(env, shape=(84, 84))
    env = gym_wrap.FrameStackObservation(env, stack_size=4)
    
    state, _ = env.reset()
    action_n = env.action_space.n
    print(f"✅ Environnement prêt. Shape: {state.shape}, Actions: {action_n}")
    
    # Agent : même stratégie d'exploration (1600 épisodes)
    agent = Agent(
        state.shape,
        action_n,
        double_q=True,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.99995,  # Identique → même durée d'exploration
        epsilon_min=0.01
    )
    print(f"🧠 Agent sur: {agent.device} | Buffer: 25k | Batch: 48 ⚡")

    episodes = 1600  # ✅ RESTE 1600 ÉPISODES
    batch_size = 48  # ⚡ Plus grand → meilleure stabilité
    timestep = 0

    episode_reward_list = []
    episode_length_list = []
    episode_epsilon_list = []
    episode_date_list = []
    episode_time_list = []

    print("\n🚀 Entraînement accéléré (1600 épisodes)...")
    print("=" * 50)

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            timestep += 1
            steps += 1
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            agent.store(state, action, reward, next_state, terminated)
            state = next_state

            if timestep % 4 == 0 and len(agent.buffer) >= batch_size:
                agent.update_net(batch_size)
            if timestep % 1500 == 0:  # ⚡ Synchronisation un peu plus fréquente
                agent.frozen_net.load_state_dict(agent.updating_net.state_dict())
            if timestep % 60000 == 0:  # ⚡ Sauvegarde moins fréquente → gain de temps
                agent.save(agent.save_dir, "LATEST_1600_FAST")

        episode_reward_list.append(total_reward)
        episode_length_list.append(steps)
        episode_epsilon_list.append(agent.epsilon)
        now = datetime.datetime.now()
        episode_date_list.append(now.strftime('%Y-%m-%d'))
        episode_time_list.append(now.strftime('%H:%M:%S'))

        if episode % 10 == 0:
            avg_reward = np.mean(episode_reward_list[-10:])
            print(f"Épisode {episode:4d} | Récompense: {avg_reward:7.2f} | ε: {agent.epsilon:.4f}")

        if episode % 100 == 0:
            agent.write_log(
                episode_date_list,
                episode_time_list,
                episode_reward_list,
                episode_length_list,
                [0.0] * len(episode_reward_list),
                episode_epsilon_list,
                f'DQN_log_1600ep_fast_ep{episode}.csv'
            )

    print("\n✅ Entraînement terminé (1600 épisodes).")
    agent.save(agent.save_dir, "DQN_1600ep_FAST_final")
    agent.write_log(
        episode_date_list,
        episode_time_list,
        episode_reward_list,
        episode_length_list,
        [0.0] * len(episode_reward_list),
        episode_epsilon_list,
        'DQN_log_1600ep_FAST_final.csv'
    )
    env.close()
    print("💾 Modèle optimisé sauvegardé.")

if __name__ == "__main__":
    main()