import gym

class SparseRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SparseRewardWrapper, self).__init__(env)
        self.env = env
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info.get("success"):
            reward = 1
        else:
            reward = 0
        return obs, reward, done, info
    
    def reset(self):
        return self.env.reset()
    
    def render(self):
        pass