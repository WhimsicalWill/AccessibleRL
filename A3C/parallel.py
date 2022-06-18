import torch.multiprocessing as mp
from networks import ActorCritic
from shared_adam import SharedAdam
from worker import worker


class ParallelEnv:
    def __init__(self, env_id, global_idx, input_shape, n_actions, num_threads):
        self.names = [str(i) for i in range(num_threads)]

        global_actor_critic = ActorCritic(input_shape, n_actions)
        global_actor_critic.share_memory()
        global_optim = SharedAdam(global_actor_critic.parameters(), lr=1e-4)

        self.threads = [mp.Process(target=worker, args=(name, input_shape, n_actions,
                        global_actor_critic, global_optim, env_id, global_idx)) for name in self.names]
        
        for thread in self.threads: # start all threads
            thread.start()

        for thread in self.threads: # join all threads
            thread.join()