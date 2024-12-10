import torch
import numpy as np
from torch.nn import MSELoss
from torch.distributions import Normal
from panda_gym.envs.core import RobotTaskEnv
# from torch.xpu import device

from model import DiscreteActor, ContinuousActor, Critic
from typing import List, Tuple
import sys
import matplotlib.pyplot as plt
import pickle


class Trainer:
    def __init__(
            self,
            env: RobotTaskEnv,
            discrete_actor: DiscreteActor,
            continuous_actor: ContinuousActor,
            critic: Critic,
            timesteps: int,
            timesteps_per_batch: int,
            max_timesteps_per_episode: int,
            training_cycles_per_batch: int = 5,
            gamma: float = 0.99,
            epsilon: float = 0.2,
            alpha: float = 3e-4,
            save_every_x_timesteps: int = 50000,
            device=None
    ):
        # Environment
        self.env = env

        # Device
        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Neural networks
        self.discrete_actor = discrete_actor.to(self.device)
        self.continuous_actor = continuous_actor.to(self.device)
        self.critic = critic.to(self.device)

        # Hypeparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

        # Iteration parameters
        self.timesteps = timesteps
        self.current_timestep = 0
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.timesteps_per_batch = timesteps_per_batch
        self.training_cycles_per_batch = training_cycles_per_batch
        self.save_every_x_timesteps = save_every_x_timesteps

        # Optimizers
        self.discrete_optimizer = torch.optim.Adam(
            params=self.discrete_actor.parameters(), lr=self.alpha)
        self.continuous_optimizer = torch.optim.Adam(
            params=self.continuous_actor.parameters(), lr=self.alpha)
        self.critic_optimizer = torch.optim.Adam(
            params=self.critic.parameters(),
                                                 lr=self.alpha)
        # Memory
        self.total_rewards: List[float] = []
        self.terminal_timesteps: List[int] = []
        self.discrete_actor_losses: List[float] = []
        self.continuous_actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.previous_print_length: int = 0
        self.current_action = "Initializing"
        self.last_save: int = 0     

    def print_status(self):
        latest_reward = 0.0
        average_reward = 0.0
        best_reward = 0.0

        latest_discrete_loss = 0.0
        avg_discrete_loss = 0.0
        latest_continuous_loss = 0.0
        avg_continuous_loss = 0.0

        latest_critic_loss = 0.0
        avg_critic_loss = 0.0
        recent_change = 0.0

        if len(self.total_rewards) > 0:
            latest_reward = self.total_rewards[-1]

            last_n_episodes = 100
            average_reward = np.mean(self.total_rewards[-last_n_episodes:])

            episodes = [
                i
                for i in range(
                    len(self.total_rewards[-last_n_episodes:]),
                    min(last_n_episodes, 0),
                    -1,
                )
            ]
            coefficients = np.polyfit(
                episodes,
                self.total_rewards[-last_n_episodes:],
                1,
            )
            recent_change = coefficients[0]

            best_reward = max(self.total_rewards)

        if len(self.discrete_actor_losses) > 0:
            avg_count = 3 * self.timesteps_per_batch
            latest_discrete_loss = self.discrete_actor_losses[-1]
            avg_discrete_loss = np.mean(
                self.discrete_actor_losses[-avg_count:])
            latest_continuous_loss = self.continuous_actor_losses[-1]
            avg_continuous_loss = np.mean(
                self.continuous_actor_losses[-avg_count:])
            latest_critic_loss = self.critic_losses[-1]
            avg_critic_loss = np.mean(self.critic_losses[-avg_count:])

        msg = f"""
            =========================================
            Timesteps: {self.current_timestep:,} / {self.timesteps:,} ({round((self.current_timestep / self.timesteps) * 100, 4)}%)
            Episodes: {len(self.total_rewards):,}
            Currently: {self.current_action}
            Latest Reward: {round(latest_reward)}
            Latest Avg Rewards: {round(average_reward)}
            Recent Change: {round(recent_change, 2)}
            Best Reward: {round(best_reward, 2)}
            Latest Discrete Actor Loss: {round(latest_discrete_loss, 4)}
            Latest Continuous Actor Loss: {round(latest_continuous_loss, 4)}
            Avg Discrete Actor Loss: {round(avg_discrete_loss, 4)}
            Avg Continuous Actor Loss: {round(avg_continuous_loss, 4)}
            Latest Critic Loss: {round(latest_critic_loss, 4)}
            Avg Critic Loss: {round(avg_critic_loss, 4)}
            =========================================
        """

        # We print to STDERR as a hack to get around the noisy pybullet
        # environment. Hacky, but effective if paired w/ 1> /dev/null
        print(msg, file=sys.stderr)

    def create_plot(self, filepath: str):
        last_n_episodes = 10

        episodes = [i + 1 for i in range(len(self.total_rewards))]
        averages = [
            np.mean(self.total_rewards[i - last_n_episodes: i])
            for i in range(len(self.total_rewards))
        ]
        trend_data = np.polyfit(episodes, self.total_rewards, 1)
        trendline = np.poly1d(trend_data)

        plt.scatter(
            episodes, self.total_rewards, color="green"
        )  # , linestyle='None', marker='o', color='green')
        plt.plot(episodes, averages, linestyle="solid", color="red")
        plt.plot(episodes, trendline(episodes), linestyle="--", color="blue")

        plt.title("Rewards per episode")
        plt.ylabel("Reward")
        plt.xlabel("Episode")
        plt.savefig(filepath)

    def save(self, directory: str):
        """
        save will save the models, state, and any additional
        data to the given directory
        """
        self.last_save = self.current_timestep

        self.discrete_actor.save(f"{directory}/discrete_actor.pth")
        self.continuous_actor.save(f"{directory}/continuous_actor.pth")
        self.critic.save(f"{directory}/critic.pth")
        self.create_plot(f"{directory}/rewards.png")

        # Now save the trainer's state data
        data = {
            "timesteps": self.timesteps,
            "current_timestep": self.current_timestep,
            "max_timesteps_per_episode": self.max_timesteps_per_episode,
            "timesteps_per_batch": self.timesteps_per_batch,
            "save_every_x_timesteps": self.save_every_x_timesteps,
            "γ": self.gamma,
            "ε": self.epsilon,
            "α": self.alpha,
            "training_cycles_per_batch": self.training_cycles_per_batch,
            "total_rewards": self.total_rewards,
            "terminal_timesteps": self.terminal_timesteps,
            "discrete_actor_losses": self.discrete_actor_losses,
            "continuous_actor_losses": self.continuous_actor_losses,
            "critic_losses": self.critic_losses,
        }
        pickle.dump(data, open(f"{directory}/state.data", "wb"))

    def load(self, directory: str):
        """
        Load will load the models, state, and any additional
        data from the given directory
        """
        # Load our models first; they're the simplest
        self.discrete_actor.load(f"{directory}/discrete_actor.pth")
        self.continuous_actor.load(f"{directory}/continuous_actor.pth")
        self.critic.load(f"{directory}/critic.pth")

        self.discrete_actor = self.discrete_actor.to(self.device)
        self.continuous_actor = self.continuous_actor.to(self.device)
        self.critic = self.critic.to(self.device)

        data = pickle.load(open(f"{directory}/state.data", "rb"))

        self.timesteps = data["timesteps"]
        self.current_timestep = data["current_timestep"]
        self.last_save = self.current_timestep
        self.max_timesteps_per_episode = data["max_timesteps_per_episode"]
        self.timesteps_per_batch = data["timesteps_per_batch"]
        self.save_every_x_timesteps = data["save_every_x_timesteps"]

        # Hyperparameters
        self.gamma = data["γ"]
        self.epsilon = data["ε"]
        self.alpha = data["α"]
        self.training_cycles_per_batch = data["training_cycles_per_batch"]

        # Memory
        self.total_rewards = data["total_rewards"]
        self.terminal_timesteps = data["terminal_timesteps"]
        self.discrete_actor_losses = data["discrete_actor_losses"]
        self.continuous_actor_losses = data["continuous_actor_losses"]
        self.critic_losses = data["critic_losses"]

        self.discrete_optimizer = torch.optim.Adam(
            self.discrete_actor.parameters(), lr=self.alpha)
        self.continuous_optimizer = torch.optim.Adam(
            self.continuous_actor.parameters(), lr=self.alpha)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.alpha)

    def run_episode(self):
        """Run a single episode."""
        observation, _ = self.env.reset()
        if isinstance(observation, dict):
            observation = observation["observation"]
            observation = torch.tensor(observation, device=self.device,
                                       dtype=torch.float32)

        timesteps = 0
        observations = []
        discrete_actions = []
        continuous_params = []
        discrete_log_probs = []
        continuous_log_probs = []
        rewards = []

        while True:
            timesteps += 1

            observations.append(observation)
            # print("observation is: ", observation)
           

            current_discrete_dist = self.discrete_actor(observation)
            current_discrete_action = current_discrete_dist.sample()
            current_discrete_log_prob = current_discrete_dist.log_prob(
                current_discrete_action).detach().cpu().numpy()
            current_discrete_action = current_discrete_action.detach(

            ).cpu().numpy()

            current_continuous_params = self.continuous_actor(observation)
            mean = current_continuous_params[current_discrete_action][
                'mean']
            std = current_continuous_params[current_discrete_action][
                'std']
            current_continuous_dist = torch.distributions.Normal(mean, std)
            current_continuous_action = current_continuous_dist.sample()
            continuous_log_prob = current_continuous_dist.log_prob(
                current_continuous_action).detach().cpu().numpy()
            current_continuous_action = current_continuous_action.detach(

            ).cpu().numpy()

            action = {
                'discrete': current_discrete_action,
                'continuous': current_continuous_action
            }

            obs, reward, terminated, _, _ = self.env.step(action)

            discrete_actions.append(current_discrete_action)
            discrete_log_probs.append(current_discrete_log_prob)

            continuous_params.append(current_continuous_action)
            continuous_log_probs.append(continuous_log_prob)

            rewards.append(reward)
            
            # Update the observation
            # observation = obs["observation"]
            # observation = torch.tensor(observation, device=self.device,
            #                            dtype=torch.float32)

            if timesteps >= self.max_timesteps_per_episode:
                terminated = True

            if terminated:
                break

        # Calculate the discounted rewards for this episode
        discounted_rewards = self.calculate_discounted_reward(rewards)

        # Get the terminal reward and record for status tracking
        self.total_rewards.append(sum(rewards))
        print("discrete_actions in episode: ", discrete_actions)
        print("continuous_params in episode: ", continuous_params)

        return (observations, discrete_actions, continuous_params,
                discrete_log_probs, continuous_log_probs, discounted_rewards)

    def rollout(self):
        """Perform a rollout of the environment and return the memory of the
        episode with the current actor models
        """
        observations = []
        discrete_log_probabilities = []
        continuous_log_probabilities = []
        discrete_actions = []
        continuous_actions = []
        rewards = []

        while len(observations) < self.timesteps_per_batch:
            self.current_action = "Rollout"
            (
                obs,
                chosen_discrete_actions,
                chosen_continuous_actions,
                discrete_log_probs,
                continuous_log_probs,
                rwds
            ) = self.run_episode()

            # Combine these arrays into overall batch
            observations += obs
            discrete_actions += chosen_discrete_actions
            continuous_actions += chosen_continuous_actions
            discrete_log_probabilities += discrete_log_probs
            continuous_log_probabilities += continuous_log_probs
            rewards += rwds

            # Increment count of timesteps
            self.current_timestep += len(obs)

            self.print_status()

        # Trim the batch memory to the batch size
        observations = observations[: self.timesteps_per_batch]
        discrete_actions = discrete_actions[: self.timesteps_per_batch]
        continuous_actions = continuous_actions[: self.timesteps_per_batch]
        discrete_log_probabilities = discrete_log_probabilities[
            : self.timesteps_per_batch]
        continuous_log_probabilities = continuous_log_probabilities[
            : self.timesteps_per_batch]
        rewards = rewards[: self.timesteps_per_batch]
        

        return (observations, discrete_actions, continuous_actions,
                discrete_log_probabilities, continuous_log_probabilities,
                rewards)

    def calculate_discounted_reward(self, rewards):
        """Calculate the discounted reward of each timestep of an episode
        given its initial rewards and episode length"""
        discounted_rewards = []
        discounted_reward = 0.0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)

        return discounted_rewards

    def calculate_normalized_advantage(self, observations, rewards):
        """Calculate the normalized advantage of each timestep of a given
        batch of episode """
        V = self.critic(observations).detach().squeeze()

        advantage = (torch.tensor(rewards, dtype=torch.float32,
                                  device=self.device)
                     - V)
        normalized_advantage = (advantage - advantage.mean()) / (
            advantage.std() + 1e-8)

        return normalized_advantage

    def training_step(
            self,
            observations,
            discrete_actions,
            continuous_actions,
            discrete_log_probabilities,
            continuous_log_probabilities,
            rewards,
            normalized_advantage
    ):
        """Peform a single epoch of training for the actors and critic model. Return the loss for each model at the end of the step"""
        # ---- Discrete Actor ----
        current_discrete_dist = self.discrete_actor(observations)
        current_discrete_log_probs = current_discrete_dist.log_prob(
            discrete_actions)
        discrete_ratio = torch.exp(
            current_discrete_log_probs - discrete_log_probabilities)
        discrete_actor_loss = -torch.min(
            discrete_ratio * normalized_advantage,
            torch.clamp(discrete_ratio, 1 - self.epsilon, 1 +
                        self.epsilon) * normalized_advantage
        ).mean()

        self.discrete_optimizer.zero_grad()
        discrete_actor_loss.backward()
        self.discrete_optimizer.step()

        # ---- Continuous Actor ----
        current_continuous_params = self.continuous_actor(observations)
        means = [current_continuous_params[
                                 int(discrete_action.item())]['mean']
                             for discrete_action in discrete_actions]
        stds = [current_continuous_params[
                     int(discrete_action.item())]['std']
                 for discrete_action in discrete_actions]

        means = torch.stack(means)
        stds = torch.stack(stds)

        current_continuous_dist = torch.distributions.Normal(means, stds)
        current_continuous_log_probs = current_continuous_dist.log_prob(
            continuous_actions)
        continuous_ratios = torch.exp(
            current_continuous_log_probs - continuous_log_probabilities)

        normalized_advantage = normalized_advantage.unsqueeze(1).unsqueeze(2)
        continuous_actor_loss = -torch.min(
            continuous_ratios * normalized_advantage,
            torch.clamp(continuous_ratios, 1 - self.epsilon,
                        1 + self.epsilon) * normalized_advantage
        ).mean()

        self.continuous_optimizer.zero_grad()
        continuous_actor_loss.backward()
        self.continuous_optimizer.step()

        # ---- Critic Network ----
        V = self.critic(observations)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32,
                                     device=self.device
                                     ).unsqueeze(-1)
        critic_loss = MSELoss()(V, reward_tensor)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return discrete_actor_loss.item(), continuous_actor_loss.item(), critic_loss.item()

    def train(self):
        while self.current_timestep <= self.timesteps:
            # Rollout to get next training batch
            observations, discrete_actions, continuous_actions, discrete_log_probabilities, continuous_log_probabilities, rewards = self.rollout()
        
            # Convert to tensors
            observations = torch.stack(observations, dim=0)
            # observations = torch.tensor(observations, dtype=torch.float32,
            #                             device=self.device)
            discrete_actions = torch.tensor(np.array(discrete_actions),
                                            dtype=torch.float32,
                                            device=self.device)
            continuous_actions = torch.tensor(np.array(continuous_actions),
                                              dtype=torch.float32,
                                              device=self.device)
            discrete_log_probabilities = torch.tensor(
                np.array(discrete_log_probabilities), dtype=torch.float32,
                device=self.device)
            continuous_log_probabilities = torch.tensor(
                np.array(continuous_log_probabilities), dtype=torch.float32,
                device=self.device)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32,
                                   device=self.device)

            # Perform training steps
            for c in range(self.training_cycles_per_batch):
                self.current_action = (
                    f"Training cycle {c+1}/{self.training_cycles_per_batch}"
                )
                self.print_status()
                # Calculate losses
                normalized_advantage = self.calculate_normalized_advantage(
                    observations, rewards)
                print("observation.shape", observations.shape)
                discrete_loss, continuous_loss, critic_loss = self.training_step(
                    observations, discrete_actions, continuous_actions, discrete_log_probabilities, continuous_log_probabilities, rewards, normalized_advantage)

                self.discrete_actor_losses.append(discrete_loss)
                self.continuous_actor_losses.append(continuous_loss)
                self.critic_losses.append(critic_loss)

            # Every x timesteps, save current status
            if self.current_timestep - self.last_save >= self.save_every_x_timesteps:
                self.current_action = "Saving"
                self.print_status()
                self.save("training")

        print("")
        print("Training complete!")
        self.save("training")
