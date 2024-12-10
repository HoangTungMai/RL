import torch
import numpy as np
from torch.nn import MSELoss
from torch.distributions import Normal
from panda_gym.envs.core import RobotTaskEnv
import random
# from torch.xpu import device

from model import DiscreteActor, ContinuousActor, Critic, QNetwork, XNetwork
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
            Decrete_action_dim: int = 3,
            Continuous_parameter_dim: int = 4,
            observation_dim: int = 31,
            device=None
    ):
        # Environment
        self.env = env

        # Device
        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.Decrete_action_dim = Decrete_action_dim
        self.Continuous_parameter_dim = Continuous_parameter_dim
        self.observation_dim  = observation_dim

        # Neural networks
        self.discrete_actor = discrete_actor.to(self.device)
        self.continuous_actor = continuous_actor.to(self.device)
        self.critic = critic.to(self.device)
        self.Q_net = QNetwork(state_dim=self.observation_dim, param_dim=self.Continuous_parameter_dim, num_actions=self.Decrete_action_dim)
        self.Q_net = self.Q_net.to(self.device)
        self.X_net = XNetwork(state_dim=self.observation_dim, param_dim=self.Continuous_parameter_dim, num_actions=self.Decrete_action_dim)
        self.X_net = self.X_net.to(self.device)
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
            params=self.critic.parameters(), lr=self.alpha)
        self.q_optimizer = torch.optim.SGD(
            params=self.critic.parameters(),lr=self.alpha)
        self.x_optimizer = torch.optim.SGD(
            params=self.critic.parameters(),lr=self.alpha)   
        # Memory
        self.total_rewards: List[float] = []
        self.terminal_timesteps: List[int] = []
        self.Q_loss: List[float] = []
        self.X_loss: List[float] = []
     
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

        if len(self.Q_loss) > 0:
            avg_count = 3 * self.timesteps_per_batch
            latest_discrete_loss = self.Q_loss[-1]
            avg_discrete_loss = np.mean(
                self.Q_loss[-avg_count:])
            latest_continuous_loss = self.X_loss[-1]
            avg_continuous_loss = np.mean(
                self.X_loss[-avg_count:])

        msg = f"""
            =========================================
            Timesteps: {self.current_timestep:,} / {self.timesteps:,} ({round((self.current_timestep / self.timesteps) * 100, 4)}%)
            Episodes: {len(self.total_rewards):,}
            Currently: {self.current_action}
            Latest Reward: {round(latest_reward)}
            Latest Avg Rewards: {round(average_reward)}
            Recent Change: {round(recent_change, 2)}
            Best Reward: {round(best_reward, 2)}
            Latest Q_loss: {round(latest_discrete_loss, 4)}
            Latest X_loss: {round(latest_continuous_loss, 4)}
            Avg Q_loss: {round(avg_discrete_loss, 4)}
            Avg X_loss: {round(avg_continuous_loss, 4)}
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
        next_observations = []
        discrete_actions = []
        continuous_params = []
        rewards = []

        while True:
                timesteps += 1

                # Record the current observation
                observations.append(observation)

                # ---- Discrete Action Selection ----
                # Compute Q-values for all discrete actions with their respective parameters
                continuous_action_params = self.X_net(observation.unsqueeze(0).to(self.device))  # Shape: (1, num_actions, param_dim)
                q_values = self.Q_net(observation.unsqueeze(0), continuous_action_params)  # Shape: (1, num_actions)

                # Select the discrete action with the highest Q-value
                if random.random() < self.epsilon:  # Exploration: Choose a random action
                    discrete_action = random.randint(0, q_values.shape[1] - 1)
                else:  # Exploitation: Choose the best action
                    discrete_action = q_values.argmax(dim=1).item()
                discrete_actions.append(discrete_action)

                # ---- Continuous Parameter Selection ----
                # Retrieve the continuous parameters for the selected discrete action
                continuous_action = continuous_action_params[0, discrete_action].detach().cpu().numpy()
                continuous_params.append(continuous_action)

                # Combine actions for the environment
                action = {
                    'discrete': discrete_action,
                    'continuous': continuous_action
                }
                

                # Step through the environment
                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                rewards.append(reward)
  

                # Store the next observation
                if isinstance(next_observation, dict):
                    next_observation = next_observation["observation"]
                next_observation = torch.tensor(next_observation, device=self.device, dtype=torch.float32)
                next_observations.append(next_observation)

                # Update the current observation
                observation = next_observation

                if terminated or truncated or timesteps >= self.max_timesteps_per_episode:
                    break
        # Calculate the discounted rewards for this episode
        discounted_rewards = self.calculate_discounted_reward(rewards)

        # Get the terminal reward and record for status tracking
        self.total_rewards.append(sum(rewards))

        return (observations,next_observations, discrete_actions, continuous_params,
                discounted_rewards)

    def rollout(self):
        """Perform a rollout of the environment and return the memory of the
        episode with the current actor models
        """
        observations = []
        discrete_actions = []
        continuous_actions = []
        rewards = []
        next_observations = []

        while len(observations) < self.timesteps_per_batch:
            self.current_action = "Rollout"
            (
                obs,
                next_obs,
                chosen_discrete_actions,
                chosen_continuous_actions,
                rwds
            ) = self.run_episode()

            # Combine these arrays into overall batch
            observations += obs
            next_observations += next_obs
            discrete_actions += chosen_discrete_actions
            continuous_actions += chosen_continuous_actions
            rewards += rwds

            # Increment count of timesteps
            self.current_timestep += len(obs)

            self.print_status()

        # Trim the batch memory to the batch size
        observations = observations[: self.timesteps_per_batch]
        next_observations = next_observations[: self.timesteps_per_batch]
        discrete_actions = discrete_actions[: self.timesteps_per_batch]
        continuous_actions = continuous_actions[: self.timesteps_per_batch]
        rewards = rewards[: self.timesteps_per_batch]

        return (observations, next_observations,
                discrete_actions, continuous_actions,
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
            next_observations,
            discrete_actions,
            continuous_actions,
            rewards,
    ):
        
        
        # ---- Compute Target Q-values ----
        with torch.no_grad():
            # Compute continuous action parameters for next state
            next_continuous_params = self.X_net(next_observations)

            # Compute Q-values for the next state
            next_q_values = self.Q_net(next_observations, next_continuous_params)

            # Max Q-value for the next state
            max_next_q_values, _ = torch.max(next_q_values, dim=1)

            # Compute the target Q-values (Bellman equation)
            target_q_values = rewards +  self.gamma * max_next_q_values
            
        #---- Q-Network Update ----
        # Compute predicted Q-values for the current state-action pairs
        continuous_action_params = self.X_net(observations)
        predicted_q_values = self.Q_net(observations, continuous_action_params)

        # Debugging

        # Gather Q-values for the selected discrete actions
        discrete_actions = discrete_actions.long()
        predicted_q_values = predicted_q_values.gather(1, discrete_actions.unsqueeze(1)).squeeze(1)

        # Compute loss for Q-network
        q_loss = torch.nn.MSELoss()(predicted_q_values, target_q_values)
        
        # Optimize Q-network
        self.q_optimizer.zero_grad()
        q_loss.backward(retain_graph = True)
        self.q_optimizer.step()
        
        # Compute the loss for X_net: negative sum of Q-values for all actions
        predicted_q_values = self.Q_net(observations, continuous_action_params)
        x_loss = -predicted_q_values.sum(dim=1).mean()
        
        # Optimize X-network
        self.x_optimizer.zero_grad()
        x_loss.backward()
        self.x_optimizer.step()

    

        return q_loss.item(), x_loss.item()

    def train(self):
        while self.current_timestep <= self.timesteps:
            # Rollout to get next training batch
            observations, next_observations, discrete_actions, continuous_actions, rewards = self.rollout()
        
            # Convert to tensors
            observations = torch.stack(observations, dim=0)
            next_observations = torch.stack(next_observations, dim=0)
            # observations = torch.tensor(observations, dtype=torch.float32,
            #                             device=self.device)
            discrete_actions = torch.tensor(np.array(discrete_actions),
                                            dtype=torch.float32,
                                            device=self.device)
            continuous_actions = torch.tensor(np.array(continuous_actions),
                                              dtype=torch.float32,
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
                
                Q_loss, x_loss,  = self.training_step(
                    observations, next_observations, discrete_actions, continuous_actions, rewards)

                self.Q_loss.append(Q_loss)
                self.X_loss.append(x_loss)

            # Every x timesteps, save current status
            if self.current_timestep - self.last_save >= self.save_every_x_timesteps:
                self.current_action = "Saving"
                self.print_status()
                self.save("training")

        print("")
        print("Training complete!")
        self.save("training")
