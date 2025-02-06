import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution, CategoricalDistribution
from gymnasium.spaces import Box
import numpy as np

class TaskEncodingNetwork(th.nn.Module):
    def __init__(self, num_tasks, embedding_dim):
        super(TaskEncodingNetwork, self).__init__()
        self.task_embedding = th.nn.Embedding(num_tasks, embedding_dim)

    def forward(self, task_id):
        embedding = self.task_embedding(task_id)
        return embedding.view(embedding.size(0), -1).float()

class CompositionalEncodingNetwork(th.nn.Module):
    def __init__(self, input_dim, task_embedding_dim, hidden_dim):
        super(CompositionalEncodingNetwork, self).__init__()
        self.fc1 = th.nn.Linear(input_dim + task_embedding_dim, hidden_dim)
        self.fc2 = th.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, obs, task_embedding):
        x = th.cat([obs, task_embedding], dim=-1).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ActorDistributionCompositionalNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, action_space, num_tasks, task_embedding_dim, hidden_dim, features_dim=256):
        super(ActorDistributionCompositionalNetwork, self).__init__(observation_space, features_dim)

        self._action_space = action_space

        input_dim = np.prod(observation_space.shape)
        self.task_encoding_net = TaskEncodingNetwork(num_tasks, task_embedding_dim)
        self.compositional_net = CompositionalEncodingNetwork(input_dim, task_embedding_dim, hidden_dim)

        if isinstance(action_space, Box):
            self._projection_net = DiagGaussianDistribution(action_space.shape[0])
        else:
            self._projection_net = CategoricalDistribution(action_space.n)

        self.optimizer = th.optim.Adam(self.parameters(), lr=3e-4)

        self.mean_net = th.nn.Linear(hidden_dim, action_space.shape[0])
        self.log_std_net = th.nn.Linear(hidden_dim, action_space.shape[0])

    def forward(self, observations, task_id):
        task_embedding = self.task_encoding_net(task_id)
        features = self.compositional_net(observations, task_embedding)
        return features

    def get_action_distribution(self, observations, task_id):
        features = self.forward(observations, task_id)
        mean = self.mean_net(features)
        log_std = self.log_std_net(features)
        log_std = th.clamp(log_std, min=-20, max=2)  # 限制 log_std 的范围
        return self._projection_net.proba_distribution(mean, log_std)

    def action_log_prob(self, observations, task_id):
        dist = self.get_action_distribution(observations, task_id)
        actions = dist.sample()
        log_prob = dist.log_prob(actions).sum(dim=-1)
        return actions, log_prob

class CompositionalCriticNetwork(th.nn.Module):
    def __init__(self, observation_space, action_space, num_tasks, task_embedding_dim, hidden_dim, features_dim=256):
        super(CompositionalCriticNetwork, self).__init__()
        self.features_dim = features_dim
        self._action_space = action_space
        input_dim = np.prod(observation_space.shape)
        action_dim = action_space.shape[0] if isinstance(action_space, Box) else action_space.n
        self.task_encoding_net = TaskEncodingNetwork(num_tasks, task_embedding_dim)
        self.compositional_net = CompositionalEncodingNetwork(input_dim, task_embedding_dim, hidden_dim)
        self.q1_net = th.nn.Sequential(
            th.nn.Linear(hidden_dim + action_dim, features_dim),
            th.nn.ReLU(),
            th.nn.Linear(features_dim, 1)
        )
        self.q2_net = th.nn.Sequential(
            th.nn.Linear(hidden_dim + action_dim, features_dim),
            th.nn.ReLU(),
            th.nn.Linear(features_dim, 1)
        )
        self.optimizer = th.optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, observations, actions, task_id):
        task_embedding = self.task_encoding_net(task_id)
        features = self.compositional_net(observations, task_embedding)
        q1_input = th.cat([features, actions], dim=-1)
        q2_input = th.cat([features, actions], dim=-1)

        q1 = self.q1_net(q1_input)
        q2 = self.q2_net(q2_input)

        return q1, q2
