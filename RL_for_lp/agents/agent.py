class AgentBase:
    def __init__(self) -> None:
        pass

    def act(self, state):
        pass

    def update(self, state, next_state, reward):
        pass  

    def train(self):
        pass

    def eval(self):
        pass