from collections import namedtuple
import random
Transition = namedtuple("Transition", ("state", "action"))

class ReplayBuffer(object):
    def __init__(self, capacity=50000, memtype=Transition):
        self.capacity = capacity
        self.memtype = memtype
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.memtype(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        del self.memory
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)