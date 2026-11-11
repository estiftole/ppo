import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from env import Environments
from model import ActorCritic

from eval import test_model

def train_PPO(device = 'cpu', lr = 2.5e-4, n_envs = 2, total_iters = 2_000, T = 128, epochs = 3, batch_size = 32*8, gamma = 0.99, gae_param = 0.95, ent_coef = 0.01, epsilon = 0.2, logging_interval = 100,):
    envs = Environments(n_envs=n_envs)
    print(f"Initiated {n_envs} environments")

    actorcritic = ActorCritic(action_space=envs.envs[0].action_space.n).to(device)

    optimizer = torch.optim.Adam(actorcritic.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1., end_factor=0., total_iters=total_iters
    )

    print(f"Started training for {total_iters} iterations")
    smoothed_rewards = [[] for _ in range(envs.len())]
    total_rewards = [[] for _ in range(envs.len())]

    for iteration in range(total_iters):
        advantages = torch.zeros((envs.len(), T), dtype=torch.float32, device=device)
        discounted_cum_rewards = torch.zeros((envs.len(), T), dtype=torch.float32, device=device)
        values = torch.zeros((envs.len(), T+1), dtype=torch.float32, device=device)
        observations = torch.zeros((envs.len(), T, 4, 84, 84), dtype=torch.float32, device=device)
        log_probs = torch.zeros((envs.len(), T), dtype=torch.float32, device=device)
        actions = torch.zeros((envs.len(), T), dtype=torch.long, device=device)
        rewards = torch.zeros((envs.len(), T), dtype=torch.float32, device=device)
        terminal = torch.zeros((envs.len(), T), dtype=torch.float16, device=device)

        for env_id in range(envs.len()):
            with torch.no_grad():
                for t in range(T):
                    obs = torch.from_numpy(envs.obs[env_id] / 255.).unsqueeze(0).float().to(device)
                    logits, value = actorcritic(obs)
                    logits, value = logits.squeeze(0), value.squeeze(0)
                    m = torch.distributions.categorical.Categorical(logits=logits)

                    if envs.done[env_id]:
                        action = torch.tensor([1]).to(device)
                    else:
                        action = m.sample()

                    log_prob = m.log_prob(action)
                    _, reward, _, _, _, done = envs.step(env_id, action)

                    observations[env_id, t] = obs
                    rewards[env_id, t] = reward
                    values[env_id, t] = value
                    log_probs[env_id, t] = log_prob
                    actions[env_id, t] = action
                    terminal[env_id, t] = done

                    if done:
                        envs.reset_env(env_id)

                total_rewards[env_id].append(envs.total_reward[env_id])
                values[env_id, T] = actorcritic(torch.from_numpy(envs.obs[env_id] / 255.).unsqueeze(0).float().to(device))[1].squeeze(0)

            for t in range(T-1, -1, -1):
                next_non_terminal = 1.0 - terminal[env_id, t]
                delta_t = rewards[env_id, t] + gamma * values[env_id, t+1] * next_non_terminal - values[env_id, t]
                cumulative_reward = rewards[env_id, t]
                if t == T-1:
                    delta_t = rewards[env_id, t] - values[env_id, t]
                    A_t = delta_t
                else:
                    A_t = delta_t + gamma * gae_param * advantages[env_id, t+1]
                    cumulative_reward += gamma * rewards[env_id, t+1]
                discounted_cum_rewards[env_id, t] = cumulative_reward
                advantages[env_id, t] = A_t


        if (iteration % logging_interval == 0) and iteration > 0:
            print(f"Iteration: {iteration}")
            for env_id in range(n_envs):
                smoothed_rewards[env_id].append(np.mean(total_rewards[env_id]))
                plt.plot(smoothed_rewards[env_id])
            total_rewards = [[] for _ in range(n_envs)]
            plt.title("Reward stats")
            plt.xlabel("Epochs")
            plt.ylabel("Average Total Reward")
            plt.savefig('assets/reward_logs.png')
            plt.close()

        dataset = DataLoader(
            TensorDataset(
                observations.reshape(-1, *observations.shape[2:]),
                values[:, :T].reshape(-1),
                log_probs.reshape(-1),
                actions.reshape(-1),
                advantages.reshape(-1),
                discounted_cum_rewards.reshape(-1),
            ), batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            for batch in dataset:
                batch_observations, batch_values, old_log_prob, actions_taken, batch_advantages, batch_discounted_cum_rewards = batch

                new_logits, value = actorcritic(batch_observations)
                new_logits, value = new_logits.squeeze(0), value.squeeze(-1)
                m = torch.distributions.categorical.Categorical(logits=new_logits)
                entropy_bonus = ent_coef * -(m.entropy()).mean()

                new_log_prob = m.log_prob(actions_taken)
                ratio = torch.exp(new_log_prob - old_log_prob)

                policy_loss_1 = ratio * batch_advantages
                policy_loss_2 = torch.clip(ratio, 1-epsilon, 1+epsilon) * batch_advantages
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                v_targ = batch_advantages + batch_values
                # v_targ = batch_discounted_cum_rewards

                value_loss1 = F.mse_loss(v_targ, value, reduction='none')
                value_loss2 = F.mse_loss(v_targ, torch.clamp(value, value - epsilon, value + epsilon),
                                            reduction='none')
                value_loss = torch.max(value_loss1, value_loss2).mean()

                loss = policy_loss + entropy_bonus + value_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actorcritic.parameters(), 0.5)
                optimizer.step()
        scheduler.step()

    print("Finished Training")

    torch.save(actorcritic.state_dict(), "checkpoints/actorcritic.pt")
    actorcritic.to(device)
    print("Saved Model")

    return actorcritic

if __name__ == "__main__":
    actorcritic = train_PPO()
    view_performance = input("Continue to view performance? ")
    if view_performance == '':
        test_model(actorcritic)
