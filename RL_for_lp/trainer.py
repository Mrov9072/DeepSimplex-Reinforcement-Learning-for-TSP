from tqdm import tqdm_notebook as tqdm
import torch
import numpy as np
from matplotlib import pyplot as plt

class Trainer:
    def __init__(self, agent, dataset, epoches = 100, train_freq = 1) -> None:
        
        self.epoches = epoches
        self.train_freq = train_freq
        
        self.agent = agent
        self.dataset = dataset

        self.metrics = {
            'train_td': [],
            'lengths': [],
            'rewards': []
        }

        self.inital_state = 0

    def train(self):
        
        self.agent.train()
        progress_bar = tqdm(range(self.epoches))

        train_trajectories = []

        for e in progress_bar:
            train_trajectory = self._roll_trajectory()
            train_trajectories.append(train_trajectory)

            if (e+1) % self.train_freq == 0:
                # Train 
                td_error_train = self._train_on_replay(train_trajectories)
                self.metrics['train_td'].append(td_error_train)
                train_trajectories = []
                
                # Eval
                self.agent.eval()
                eval_trajectory = self._roll_trajectory()
                self.metrics['rewards'].append( np.mean(eval_trajectory['rewards']) )
                self.metrics['lengths'].append( np.mean(eval_trajectory['length']) )
                self.agent.train()

    def evaluate(self):
        pass

    def get_errors(self):
        return self.errors

    def plot_error(self):
        pass

    def get_best_trajectory(self):
        self.agent.eval()
        return self._roll_trajectory()
    
    def plot_best(self):
        traj = self.get_best_trajectory()['states']
        coords = self.dataset.coord_matrix
        plt.scatter(coords[:,0], coords[:,1])
        for i in range(len(traj)-1):
            plt.plot([coords[traj[i],0],coords[traj[i+1],0]],[coords[traj[i],1],coords[traj[i+1],1]])

    def _train_on_replay(self, trajectoies):
        td_errors = []
        for traj in trajectoies:
            states = traj['states'][:-1]
            rewards = traj['rewards']
            a_actions = traj['available_actions']
            for i in range(1, len(states)-1):
                td = self.agent.update(states[i-1], states[i], rewards[i], a_actions[i])
                td_errors.append(td)
        
        return np.mean(td_errors)

    def _roll_trajectory(self):
        available_actions = list(range(self.dataset.get_dim()))
        state = self.inital_state
        traj = [state]
        rewards = []
        a_actions = []
        length = 0
        while len(available_actions) > 1:
            available_actions.remove(state)
            a_actions_t = torch.tensor(available_actions).long()
            next_state = self.agent.act(state, a_actions_t)
            reward = self._get_reward(state, next_state)
            
            traj.append(next_state.item())
            rewards.append(reward)
            a_actions.append(a_actions_t)
            length += self.dataset.get_distance(state, next_state)

            state = next_state

        available_actions.remove(state)
        traj.append(self.inital_state)
        a_actions.append(torch.tensor(available_actions).long())
        rewards.append(self._get_reward(traj[-2], traj[-1]))
        length += self.dataset.get_distance(state, next_state)
        return {
            'states': traj, 
            'rewards': rewards,
            'length': length,
            'available_actions': a_actions
        }

    def _get_reward(self, s1, s2):
        return 1 + self.dataset.get_max_distance() - self.dataset.get_distance(s1, s2)