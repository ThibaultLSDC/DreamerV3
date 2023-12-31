import numpy as np
import unittest
from DreamerV3.buffer import Episode, Buffer


class TestEpisode(unittest.TestCase):
    def test_episode_shape(self):
        max_len = 10
        dummy_state = np.array([2, 3])
        dummy_action = np.array([1, 2])
        dummy_reward = np.array([0.1])
        dummy_done = np.array([False])
        structure = {'state': dummy_state, 'action': dummy_action, 'reward': dummy_reward, 'done': dummy_done}
        episode = Episode(max_len, structure)
        
        for i in range(max_len):
            episode.push(structure)

        ep = episode.flush()
        self.assertEqual(ep['state'].shape, (max_len, *dummy_state.shape))
        self.assertEqual(ep['action'].shape, (max_len, *dummy_action.shape))
        self.assertEqual(ep['reward'].shape, (max_len, *dummy_reward.shape))
        self.assertEqual(ep['done'].shape, (max_len, *dummy_done.shape))
        self.assertEqual(ep['first'].shape, (max_len, *dummy_reward.shape))


class TestBuffer(unittest.TestCase):
    def test_buffer_shape(self):
        max_len = 10
        dummy_state = np.array([2, 3])
        dummy_action = np.array([1, 2])
        dummy_reward = np.array([0.1])
        dummy_done = np.array([False])
        structure1 = {'state': dummy_state, 'action': dummy_action, 'reward': dummy_reward, 'done': dummy_done}
        structure2 = {k: v+1 if k != 'done' else v for k, v in structure1.items()}
        structure3 = {k: v+2 if k != 'done' else v for k, v in structure1.items()}
        ep = Episode(max_len, structure1)

        buffer_structure = structure1 | {'first': dummy_done}

        buffer = Buffer(max_len*2, buffer_structure)

        for i in range(int(2.5*max_len)):
            if i < max_len:
                ep.push(structure1)
            elif i < 2*max_len:
                ep.push(structure2)
            else:
                ep.push(structure3)
            if len(ep) == max_len:
                buffer.push(ep.flush())
        buffer.push(ep.flush())
        
        batch = buffer.sample(2, 4)
        self.assertEqual(batch['state'].shape, (2, 5, *dummy_state.shape))
        self.assertEqual(batch['action'].shape, (2, 5, *dummy_action.shape))
        self.assertEqual(batch['reward'].shape, (2, 5, *dummy_reward.shape))
        self.assertEqual(batch['done'].shape, (2, 5, *dummy_done.shape))
        self.assertEqual(batch['first'].shape, (2, 5, *dummy_reward.shape))

        res_state = np.stack([dummy_state for _ in range(20)], axis=0)
        res_state[:5] += 2
        res_state[10:] += 1

        self.assertTrue(np.all(buffer.buffer['state'] == res_state))


if __name__ == '__main__':
    unittest.main()
