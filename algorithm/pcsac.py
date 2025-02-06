from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, Schedule,MaybeCallback
from stable_baselines3.common.utils import get_parameters_by_name
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from algorithm.network import ActorDistributionCompositionalNetwork, CompositionalCriticNetwork
from algorithm.buffer import CustomReplayBuffer

SelfSAC = TypeVar("SelfSAC", bound="PCSAC")

class PCSAC(OffPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: ActorDistributionCompositionalNetwork
    critic: CompositionalCriticNetwork
    critic_target: CompositionalCriticNetwork


    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = CustomReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        num_tasks: int = 4,
        task_embedding_dim: int = 10,
        hidden_dim: int = 256
    ):
        self.num_tasks = num_tasks
        self.task_embedding_dim = task_embedding_dim
        self.hidden_dim = hidden_dim

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

        if self.target_entropy == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            self.target_entropy = float(self.target_entropy)
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
            self.ent_coef_tensor = self.log_ent_coef.exp().detach()
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        features_dim = 256
        self.actor = ActorDistributionCompositionalNetwork(
            self.observation_space,
            self.action_space,
            self.num_tasks,
            self.task_embedding_dim,
            self.hidden_dim,
            features_dim=features_dim
        )
        self.critic = CompositionalCriticNetwork(
            self.observation_space,
            self.action_space,
            self.num_tasks,
            self.task_embedding_dim,
            self.hidden_dim,
            features_dim=features_dim
        )
        self.critic_target = CompositionalCriticNetwork(
            self.observation_space,
            self.action_space,
            self.num_tasks,
            self.task_embedding_dim,
            self.hidden_dim,
            features_dim=features_dim
        )

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, done, infos):
        """
        Store transition in the replay buffer.
        :param replay_buffer: Replay buffer object
        :param buffer_action: Action to store
        :param new_obs: New observation
        :param reward: Reward for the transition
        :param done: Is the episode done?
        :param infos: Extra information about the transition
        """
        if not isinstance(infos, list):
            raise ValueError(f"Expected infos to be a list of dictionaries, but got: {infos}")

        for idx, info in enumerate(infos):
            task_id = info.get('task_id', 0)


            replay_buffer.add(self._last_obs[idx], new_obs[idx], buffer_action[idx], reward[idx], done[idx], infos,
                               task_id=task_id)

        self._last_obs = new_obs

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if self.ent_coef_optimizer is None:
            if self.ent_coef_tensor is None:
                self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer]
        optimizers = [opt for opt in optimizers if opt is not None]
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size)

            if self.use_sde:
                self.actor.reset_noise(self.batch_size)

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data[0].next_observations,
                                                                         replay_data[1])
                next_q_values = th.cat(
                    self.critic_target(replay_data[0].next_observations, next_actions, replay_data[1]), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data[0].rewards + (1 - replay_data[0].dones) * self.gamma * (
                        next_q_values - self.ent_coef_tensor * next_log_prob)

            current_q_values = self.critic(replay_data[0].observations, replay_data[0].actions, replay_data[1])

            q1, q2 = current_q_values
            critic_loss = (th.nn.functional.mse_loss(q1, target_q_values) +
                           th.nn.functional.mse_loss(q2, target_q_values)) / 2

            actor_actions, log_prob = self.actor.action_log_prob(replay_data[0].observations, replay_data[1])
            q_values_pi = th.cat(self.critic(replay_data[0].observations, actor_actions, replay_data[1]), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (self.ent_coef_tensor * log_prob - min_qf_pi).mean()

            if self.ent_coef_optimizer is not None:
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                self.ent_coef_tensor = self.log_ent_coef.exp().detach()
                ent_coef_losses.append(ent_coef_loss.item())
                ent_coefs.append(self.ent_coef_tensor.item())

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/entropy_loss", np.mean(ent_coef_losses))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
    def learn(
            self: SelfSAC,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "PCSAC",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "critic_target"]
        return state_dicts, []
