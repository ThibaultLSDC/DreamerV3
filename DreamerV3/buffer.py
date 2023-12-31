import numpy as np


class Episode:
    def __init__(self, max_len, structure):
        self.max_len = max_len
        self.structure = structure
        self.reset()
    
    def reset(self):
        self.episode = {key: [] for key in self.structure}
    
    def push(self, data):
        for key in self.structure:
            self.episode[key].append(data[key])
    
    def flush(self):
        episode = self.episode
        self.reset()
        for key in self.structure:
            if isinstance(episode[key][0], np.ndarray):
                episode[key] = np.stack(episode[key], axis=0)
            else:
                episode[key] = np.array(episode[key])
        episode['first'] = np.zeros(episode['reward'].shape, dtype=bool)
        episode['first'][0] = True
        episode['length'] = episode['reward'].shape[0]
        return episode
    
    def __len__(self):
        return len(self.episode['state'])
    

class Buffer:
    def __init__(self, max_len, structure):
        self.max_len = max_len
        self.structure = structure
        self.reset()

        self.idx = 0
    
    def reset(self):
        self.buffer = {key: np.zeros((self.max_len, *self.structure[key].shape), dtype=self.structure[key].dtype) for key in self.structure}
    
    def push(self, episode):
        l = episode['length']
        i = self.idx
        m = self.max_len
        for key in self.structure:
            if l + i < m:
                self.buffer[key][i:i+l] = episode[key]
            else:
                self.buffer[key][i:] = episode[key][:m-i]
                self.buffer[key][:l+i-m] = episode[key][m-i:]
        self.idx = (i + l) % m
        
    def sample(self, batch_size, sequence_len):
        idx = np.random.randint(0, self.max_len - sequence_len - 1, batch_size)
        batch = {}
        for key in self.buffer:
            batch[key] = self.buffer[key][idx[:, None] + np.arange(sequence_len+1)]
        return batch
