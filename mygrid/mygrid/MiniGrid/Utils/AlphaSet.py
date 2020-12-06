import numpy as np
from sklearn.neighbors import KDTree


class AlphaSet(object):
    """ AlphaSet

    Data structure for the memory pool of Gaussian Process. In the AlphaSet the `new_value = alpha * old_value + (1-alpha) * new_value`
    """
    def __init__(self, func=lambda new, old: 0.5*old + 0.5*new, capacity=500, dtype=np.float64):
        super().__init__()
        self.func = func
        self.dtype = dtype
        self.keys = []
        self.values = []

        self.old_keys = []
        self.old_values = []

        self.capacity = capacity
        self.old_position = 0
        self.position = 0

    def add(self, _key, _value):
        """ Add elements into the AlphaSet.

        The (key, value) pair is (parameter, score), in order to speed up the search, we transfrom the paramter into bytes.
        """
        assert _key.dtype == self.dtype, "Unmatch dtype in AlphaSet: {}, {}".format(_key.dtype, self.dtype)
        _key = _key.tobytes()

        if _key in self.keys:
            victim = self.keys.index(_key)
        else:
            victim = self.find_victim()
            self.values[victim] = []
        
        self.keys[victim] = _key
        self.values[victim].append(_value)

    def find_victim(self):
        """ Find the next position to insert elements.
        """
        if len(self.keys) < self.capacity:
            self.keys.append(None)
            self.values.append(None)
        victim = self.position
        self.position = (self.position + 1) % self.capacity
        return victim

    def query_nn(self, data, tree_data):
        """ Query the neareast neighbor. (Euclidean distance)
        """
        if len(self.old_keys) > 0:
            tree = KDTree(np.array(tree_data), leaf_size=2)                    
            dist, ind = tree.query(data.reshape(1, -1), k=1)
            return np.array(self.old_values)[ind][0][0] 
        else:
            return np.float64(2.0)

    def get_data(self):
        """ To get the relative improvement from the AlphaSet.

        Calculate the relative improvement, i.e. (parameter, new_score - relative)
        """
        _keys = [np.frombuffer(_key, dtype=self.dtype) for _key in self.keys]
        _old_keys = [np.frombuffer(_key, dtype=self.dtype) for _key in self.old_keys]

        _values = [self.values[i] - self.old_values[self.old_keys.index(self.keys[i])] if (self.keys[i] in self.old_keys) else 0.0 for i in range(len(self.values))]

        # Uncomment this if you want to use relative improvement w.r.t. the performance compared to the score of the most similar parameter. (in Euclidean distance)
        # _values = [self.values[i] - self.old_values[self.old_keys.index(self.keys[i])] if (self.keys[i] in self.old_keys) else self.values[i] - self.query_nn(np.frombuffer(self.keys[i], dtype=self.dtype), _old_keys) for i in range(len(self.values))]
        
        return (_keys.copy(), _values.copy())

    def calculate_mean(self):
        self.values = [np.mean(_values) for _values in self.values]

    def save_status(self):
        """ Save (key, value) to (old_key, old_value)
        """
        for i in range(len(self.keys)):
            if self.keys[i] in self.old_keys:
                __old_values = self.old_values[self.old_keys.index(self.keys[i])]
                self.old_values[self.old_keys.index(self.keys[i])] = self.func(__old_values, self.values[i])
            else:
                if len(self.old_keys) < self.capacity:
                    self.old_keys.append(None)
                    self.old_values.append(None)
                self.old_keys[self.old_position] = self.keys[i]
                self.old_values[self.old_position] = self.values[i]
                self.old_position = (self.old_position + 1) % self.capacity

        self.keys = []
        self.values = []
        self.position = 0

    def __getitem__(self, i):
        _key = self.keys[i]
        _value = self.values[i]
        _key = np.frombuffer(_key, dtype=self.dtype)
        return (_key, _value)
    
    def __len__(self):
        return len(self.keys)










