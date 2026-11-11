from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4), nn.Tanh(),
            nn.Conv2d(16, 32, 4, stride=2), nn.Tanh(),
            nn.Flatten(), nn.Linear(2592, 256), nn.Tanh(),)
        self.actor = nn.Sequential(nn.Linear(256, action_space))
        self.critic = nn.Sequential(nn.Linear(256, 1))

    def forward(self, x):
        h = self.head(x)
        return self.actor(h), self.critic(h)
