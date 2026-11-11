import torch

from env import Environments
from model import ActorCritic
from gymnasium.wrappers import RecordVideo


def test_model(actorcritic, device='cpu'):
    base_env = Environments(render_mode='rgb_array')

    wrapped_env = RecordVideo(
        base_env.envs[0],
        video_folder="./assets",
        episode_trigger=lambda id: True,
        name_prefix="play"
    )

    obs, _ = wrapped_env.reset()
    for _ in range(1000):
        obs_tensor = torch.from_numpy(obs / 255.).unsqueeze(0).float().to(device)
        logits, _ = actorcritic(obs_tensor)
        m = torch.distributions.categorical.Categorical(logits=logits)
        action = m.sample().item()

        obs, _, terminated, truncated, _ = wrapped_env.step(action)
        if terminated or truncated:
            break

    wrapped_env.close()


if __name__ == "__main__":
    dummy_env = Environments().envs[0]
    actorcritic = ActorCritic(action_space=dummy_env.action_space.n)
    actorcritic.load_state_dict(torch.load("checkpoints/actorcritic.pt", weights_only=True))
    actorcritic.eval()

    test_model(actorcritic)
