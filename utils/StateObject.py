# Create a class for storing state info
class State:
    def __init__(self, x, u, p):
        self.x = x
        self.u = u
        self.p = p
    
    def get_state(self):
        return self.x
    
    def get_action(self):
        return self.u
    
    def get_belief(self):
        return self.p
    
    def set_state(self, x):
        self.x = x
    
    def set_action(self, u):
        self.u = u
    
    def set_belief(self, p):
        self.p = p
    
    def __repr__(self):
        return "State()"
    
    def __str__(self):
        return f'Action: {self.u}, State: {self.x}, Belief: {self.p}'


