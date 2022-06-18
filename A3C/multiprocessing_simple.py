import os
import torch.multiprocessing as mp

def worker(name):
    print(f"hello {name}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    process = mp.Process(target=worker, args=('dale',))
    process.start()
    process.join()