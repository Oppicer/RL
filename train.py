import torch
import torch.nn as nn
import torch.optim as optim

# Simple environment with a binary action space.
# Each episode is a single step where action 1 yields +1 reward
# and action 0 yields -1 reward. This keeps the RL loop concise
# while still allowing us to demonstrate training logic.
class SimpleEnv:
    def reset(self):
        self.state = torch.tensor([[0.0]])
        return self.state

    def step(self, action: int):
        reward = 1.0 if action == 1 else -1.0
        done = True  # one-step episode
        return self.state, reward, done, {}


# Policy network mapping the state to action logits.
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 2)

    def forward(self, x):
        return self.fc(x)


# Generative reward model predicting reward from state-action pairs.
class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, state, action_one_hot):
        # concatenate state and one-hot action encoding
        x = torch.cat([state, action_one_hot], dim=-1)
        return self.fc(x)


def train(num_episodes: int = 1000, log_path: str = "training.log"):
    """Minimal training loop demonstrating RL with a reward model."""
    env = SimpleEnv()
    policy = PolicyNet()
    reward_model = RewardModel()

    policy_optim = optim.Adam(policy.parameters(), lr=1e-2)
    reward_optim = optim.Adam(reward_model.parameters(), lr=1e-2)

    logs = []
    for episode in range(num_episodes):
        state = env.reset()
        logits = policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        action_one_hot = nn.functional.one_hot(action, num_classes=2).float()

        next_state, reward, done, _ = env.step(action.item())
        reward_tensor = torch.tensor([[reward]])

        # Train the reward model on actual environment reward
        pred_reward = reward_model(state, action_one_hot)
        loss_r = nn.functional.mse_loss(pred_reward, reward_tensor)
        reward_optim.zero_grad()
        loss_r.backward()
        reward_optim.step()

        # Update policy using the predicted reward (generative reward modeling)
        log_prob = dist.log_prob(action)
        policy_loss = -log_prob * pred_reward.detach()
        policy_optim.zero_grad()
        policy_loss.backward()
        policy_optim.step()

        logs.append({"episode": episode, "reward": reward, "pred_reward": pred_reward.item()})

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: reward={reward:+.1f} predicted={pred_reward.item():+.2f}")

    with open(log_path, "w") as f:
        for item in logs:
            f.write(f"{item['episode']},{item['reward']},{item['pred_reward']}\n")


if __name__ == "__main__":
    train()
