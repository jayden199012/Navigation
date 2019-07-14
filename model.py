import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from utils import parse_params
from collections import namedtuple, deque
import numpy as np
import random
import sys


class Model(nn.Module):
    def __init__(self, action_size, state_size, hidden_layer_dim, num_layers,
                 seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size
        self.hidden_layer_dim = hidden_layer_dim
        self.num_layers = num_layers
        self.module_list = self.create_ddqn()

    def create_ddqn(self):
        module_list = nn.ModuleList()
        input_size = self.state_size
        for i in range(self.num_layers):
            module = nn.Sequential()
            ln = nn.Linear(input_size, self.hidden_layer_dim)

            # Each 'layer' consists of a fc layer, batchnorm layer and relu
            module.add_module(f"fc_layer_{i}", ln)
            module.add_module(f"batch_norm_{i}",
                              nn.BatchNorm1d(self.hidden_layer_dim))
            module.add_module(f"relu_{i}", nn.LeakyReLU())

            # Add layer to module list
            module_list.append(module)
            input_size = self.hidden_layer_dim
        module_list.append(nn.Linear(self.hidden_layer_dim, self.action_size))
        return module_list

    def forward(self, x):
        y_pred = x
        for i in range(len(self.module_list)):
            y_pred = self.module_list[i](y_pred)
        return y_pred


class Agent():
    def __init__(self, param_dir, action_size, state_size):
        self.params = parse_params(param_dir)
        self.action_size = action_size
        self.state_size = state_size
        self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        # Initiate local model
        self.local_model = Model(self.action_size, self.state_size,
                                 self.params['hidden_layer_dim'],
                                 self.params['num_layers'],
                                 self.params['seed']).to(self.device)

        # Initiate Target model
        self.target_model = Model(self.action_size, self.state_size,
                                  self.params['hidden_layer_dim'],
                                  self.params['num_layers'],
                                  self.params['seed']).to(self.device)
        self.memories = ReplayBuffer(self.params['buffer_size'],
                                     self.action_size,
                                     self.params['batch_size'],
                                     self.params['seed'],
                                     self.device)
        self.optimizer = optim.Adam(self.local_model.module_list.parameters(),
                                    lr=self.params['learning_rate'])
        self.time_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memories.add(state, action, reward, next_state, done)
        self.time_step += 1
        if self.time_step % self.params['update_freq']:
            if len(self.memories) > self.params['batch_size']:
                experiences = self.memories.sample()
                self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # we use detach for target model as we do not want to pass gradient
        # back for these two operations

        # get best action index for next state from local model
        ts_next_best_index = torch.argmax(
                self.local_model(next_states).detach(), dim=1, keepdim=True)

        # get best action value for next state from target model using local
        # index
        ts_val_next = self.target_model(next_states).detach().gather(
                1, ts_next_best_index)
        local = self.local_model(states).gather(1, actions)

        # no rewards with episode ends
        tagret = rewards + ts_val_next * self.params['gamma'] * (1 - dones)
        loss = F.mse_loss(local, tagret)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_udpate()

    def soft_udpate(self):
        for tp, lp in zip(self.target_model.module_list.parameters(),
                          self.local_model.module_list.parameters()):
            tp.data.copy_(self.params['TAU']*lp.data +
                          (1.0-self.params['TAU'])*tp.data)

    def act(self, state, eps):
        self.local_model.eval()
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            actions = self.local_model(state)
        self.local_model.train()
        if random.random() > eps:
            action = torch.argmax(actions, dim=1).cpu().numpy()[0]
        else:
            action = np.random.randint(self.action_size)
        return action

    def train(self, env, eps_start, eps_decay, eps_min,
              score_window_size=100):
        brain_name = env.brain_names[0]
        eps = eps_start
        scores = []
        scores_window = deque(maxlen=score_window_size)
        for e in range(1, self.params['n_episodes'] + 1):
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0
            for t in range(self.params['max_t']):
                action = self.act(state, eps)
                env_info = env.step(action)[brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                self.step(state, action, reward, next_state, done)
                score += reward
                state = next_state
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            eps = max(eps_min, eps_decay*eps)
            if e % 100 == 0:
                print(f"Episode  {e}: Average score: {np.mean(scores_window)}")
                sys.stdout.flush()
            if np.mean(scores_window) >= self.params['pass_score']:
                print(f"Environment solved in Episode  {e}")
                print(f"Average score: {np.mean(scores_window)}")
                torch.save(self.local_model.state_dict(),
                           self.params['working_dir'])
                break
        return scores


class ReplayBuffer():
    def __init__(self, buffer_size, action_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        # create a named tuple object to store training samples
        self.experience = namedtuple("Experience",
                                     field_names=['state', "action", "reward",
                                                  "next_state", "done"])
        self.batch_size = batch_size
        self.device = device
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        '''
            create a new namedtuple for each experience and append it to memory
            All inputs are in numpy format
        '''
        self.memory.append(self.experience(
                state, action, reward, next_state, done))

    def sample(self):
        sampled_exp = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(
            np.vstack([e.state for e in sampled_exp if e is not None])
            ).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in sampled_exp if e is not None])
            ).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in sampled_exp if e is not None])
            ).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in sampled_exp if e is not None])
            ).float().to(self.device)
        dones = torch.from_numpy(
                 np.vstack(
                         [e.done for e in sampled_exp if e is not None]
                             ).astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

