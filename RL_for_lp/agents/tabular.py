import torch
from .agent import AgentBase

class TExplorationStrategy:
    def __init__(self):
        pass

    def sample_action(self, q_values, possible_actions):
        pass

    def feedback(self, reward):
        pass

class TEpsGreedy(TExplorationStrategy):
    def __init__(self, epsilon = 1.0, n_epoch = 10_000):
        self.eps = epsilon
        self.n_epoch = n_epoch
        self.i = 0

    def sample_action(self, q_values, possible_actions):

        self.eps -= self.i/self.n_epoch
        self.i+=1

        if torch.rand(1).item() < self.eps:
            idx = torch.randint(0, possible_actions.shape[0], (1,))[0]
            return possible_actions[idx]
        else:
            return possible_actions[torch.argmax(q_values[possible_actions])]
        
class TBolzmanSampling(TExplorationStrategy):
    def __init__(self, temperature = 0.1):
        self.temperature = temperature

    def sample_action(self, q_values, possible_actions):
        distribution = torch.softmax( q_values[possible_actions] / self.temperature, 0 )
        sample_idx = torch.distributions.Categorical(distribution).sample()
        return possible_actions[sample_idx]

class ThompsonSampling(TExplorationStrategy):
    def __init__(self, n):
        self.alphas = torch.zeros( n )
        self.betas = torch.zeros( n )

        self._last_act = 0

    def sample_action(self, q_values, possible_actions):
        alphas = q_values[possible_actions]
        alphas = alphas / (alphas.sum()+1e-5)
        betas = 1.0 - alphas
        samples = torch.distributions.beta.Beta(alphas + 1e-2, betas).sample()
        max_idx = torch.argmax(samples)
        self._last_act = possible_actions[max_idx]
        return self._last_act
    
    def feedback(self, reward):
        self.alphas[self._last_act] += reward
        self.betas[self._last_act] += 1.0 - reward

class TQAgent(AgentBase):
    def __init__(self, dim, gamma = 0.9, alpha = 0.1, sampling_strategy = TEpsGreedy(epsilon=0.1)) -> None:
        self.n = dim
        self.gamma = gamma
        self.alpha = alpha
        self.sampling_strategy = sampling_strategy

        self.train_mode = True
        self.q_table = torch.zeros(self.n, self.n) - int(1e-5)

    def act(self, state, possible_actions):
        q_values = self.q_table[state]
        if self.train_mode:
            return self.sampling_strategy.sample_action(q_values, possible_actions)
        else:
            idx = torch.argmax(q_values[possible_actions])
            return possible_actions[idx]

    def _get_max_q(self, state, available_actions):
        #return self.q_table[state][available_actions].max()
        return self.q_table[state][:].max()
    
    def _get_actual_q(self, state, next_state):
        return self.q_table[state][next_state]

    def update(self, state, next_state, reward, available_actions):
        
        self.sampling_strategy.feedback(reward)

        next_q = self._get_max_q(state, available_actions)
        current_q = self._get_actual_q(state, next_state)

        td_error = reward + self.gamma * next_q - current_q
        self.q_table[state][next_state] += self.alpha * td_error
        
        return td_error

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False