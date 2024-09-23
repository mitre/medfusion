
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DistributedSampler

import torch.multiprocessing as mp 
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler



class DistributedWeightedSampler(DistributedSampler):
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True, seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.rank)
        
        # Generate a list of indices based on the weights
        full_indices = torch.multinomial(
            self.weights, len(self.dataset), self.replacement, generator=g
        ).tolist()

        # Partition the indices based on rank
        indices_per_replica = len(full_indices) // self.num_replicas
        start = self.rank * indices_per_replica
        end = start + indices_per_replica if self.rank != self.num_replicas - 1 else len(full_indices)
        
        return iter(full_indices[start:end])

    def __len__(self):
        return len(self.dataset) // self.num_replicas
    
    
class SimpleDataModule(pl.LightningDataModule):

    def __init__(self,
                 ds_train: object,
                 ds_val:object =None,
                 ds_test:object =None,
                 batch_size: int = 1,
                 num_workers: int = mp.cpu_count(),
                 seed: int = 1, 
                 pin_memory: bool = False,
                 weights: list = None 
                ):
        super().__init__()
        self.hyperparameters = {**locals()}
        self.hyperparameters.pop('__class__')
        self.hyperparameters.pop('self')

        self.ds_train = ds_train 
        self.ds_val = ds_val 
        self.ds_test = ds_test 

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed 
        self.pin_memory = pin_memory
        self.weights = weights

   

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        if self.weights is not None:
            sampler = DistributedWeightedSampler(self.ds_train, self.weights, replacement=True, seed=self.seed)
        else:
            sampler = DistributedSampler(self.ds_train)
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers, 
                          sampler=sampler, generator=generator, drop_last=True, pin_memory=self.pin_memory)


    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_val is not None:
            return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, 
                                generator=generator, drop_last=False, pin_memory=self.pin_memory)
        else:
            raise AssertionError("A validation set was not initialized.")


    def test_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_test is not None:
            return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, 
                            generator = generator, drop_last=False, pin_memory=self.pin_memory)
        else:
            raise AssertionError("A test test set was not initialized.")

   
   
    


        
      


    