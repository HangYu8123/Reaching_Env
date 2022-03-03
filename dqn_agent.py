import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
import csv
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 0.005              # learning rate 
UPDATE_EVERY = 2        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, learning_rate =LR):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """


        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, record = False,filename="buffer"):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, record, filename)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
        # if record:
        #     self.record()
    def active_learn(self, batch_size = BATCH_SIZE):
        #print(len(self.memory))
        if len(self.memory) > batch_size:
            experiences = self.memory.sample(batch_size)
            self.learn(experiences, GAMMA)
        # if record:
        #     self.record()
    def record(self, state, action, reward, next_state, done, filename = "DQN_data.csv"):
        pass
    def read_data(self, buffer,filename = "DQN_data.csv"):
        pass


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            prob = torch.nn.functional.softmax(action_values.cpu().data.detach(),dim=1)
            prob=prob.numpy()
            prob=prob[0]
            return prob
        else:
            return np.ones(self.action_size)/self.action_size

    def act_ps(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            prob = action_values.cpu().data.detach()
            prob = prob.numpy()
            prob = prob[0]
            return prob
        else:
            return np.ones(self.action_size) / self.action_size

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
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
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, save = False,filename="buffer"):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        #print(e)
        self.memory.append(e)
        if save:
            self.save_one( state, action, reward, next_state, done, filename)
    
    def sample(self, batch_size = BATCH_SIZE):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k = batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def save(self, state, action, reward, next_state, done, filename="buffer"):
        with open(filename + ".csv", 'w') as csvfile:
            kf_writer = csv.writer(csvfile, delimiter=' ',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(self.obs)):
                #print(self.obs[i].numpy().tolist() , self.action[i].numpy().tolist() , self.reward[i].numpy().tolist() , self.next_obs[i].numpy().tolist())
                if done:
                    done = 1
                else:
                    done =0
                temp = state.tolist() + action.tolist() + [reward] + next_state.tolist() +  [done]

                #kf_writer.writerow( [ self.obs[i].numpy(), self.action[i].numpy(), self.reward[i].numpy() ,self.next_obs[i].numpy(),  self.done[i] ] )
                kf_writer.writerow( temp)
                #print("done for once")

    def save_one(self, state, action, reward, next_state, done, filename="buffer"):
        with open(filename + ".csv", 'a') as csvfile:
            kf_writer = csv.writer(csvfile, delimiter=' ',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if done:
                done = 1
            else:
                done =0
            #print (state.tolist() , action.tolist() , [reward] , next_state.tolist() ,  [done])
            temp = state.tolist() + [action] + [reward] + next_state.tolist() +  [done]

                #kf_writer.writerow( [ self.obs[i].numpy(), self.action[i].numpy(), self.reward[i].numpy() ,self.next_obs[i].numpy(),  self.done[i] ] )
            kf_writer.writerow( temp)
                #print("done for once")
    def read(self, filename="buffer", obs_size = 4, action_size = 1):
        with open(filename+ ".csv", 'rb') as file:
            reader = csv.reader(file, 
                            quoting = csv.QUOTE_ALL,
                            delimiter = ' ')
        
        # storing all the rows in an output list
            output = []
            for row in reader:
                row = map(float, row)
                if row[-1] == 1:
                    done_ = True
                else:
                    done_ = False
                obs_ = row[:obs_size]
                action_ = row[obs_size: obs_size+action_size]
                reward_ = row[obs_size+action_size]
                next_obs_ = row[obs_size+action_size+1: 1 + obs_size+action_size + obs_size ]
                #done_ = row[-1]
                self.add(obs_, int(action_[0]), reward_, next_obs_, done_)
                #print(type(obs_[0]))
                #output.append(row[:])
        #print(output)
        return output

    def read_plus_fb(self, feedbacks,filename="buffer", obs_size=4, action_size=1):
        with open(filename + ".csv", 'rb') as file:
            reader = csv.reader(file,
                                quoting=csv.QUOTE_ALL,
                                delimiter=' ')

            # storing all the rows in an output list
            output = []
            cnt = 0
            print(reader)
            for row in reader:
                row = map(float, row)
                if row[-1] == 1:
                    done_ = True
                else:
                    done_ = False
                obs_ = row[:obs_size]
                action_ = row[obs_size: obs_size + action_size]
                reward_ = row[obs_size + action_size]
                next_obs_ = row[obs_size + action_size + 1: 1 + obs_size + action_size + obs_size]
                # done_ = row[-1]
                #
                #print(len(reader))
                self.add(obs_, int(action_[0]), feedbacks[cnt], next_obs_, done_)
                cnt += 1
                print(feedbacks)
                print(cnt)
                # output.append(row[:])
        # print(output)
        return output