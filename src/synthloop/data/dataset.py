from datasets import load_dataset, load_from_disk
from synthloop.config import authenticate_hf
import os

class BaseDataset:
    def __init__(self, config: dict):
        self.config = config
        self.dataset = None
        self.cache_dir = config['data']['cache_dir']
        self.hf_path = config['data']['dataset']
    
    def load(self):
        authenticate_hf()
        size = self.config['data'].get('size', None)
        split = self.config['data'].get('split', 'train')
        
        if size:
            split = f"{split}[:{size}]"
            
        self.dataset = load_dataset(
            self.hf_path,
            split=split,
            cache_dir=self.cache_dir
        )

    def select_columns(self, columns: list):
        self.dataset = self.dataset.select_columns(columns)
        return self
    
    def rename_columns(self, mapping: dict):
        self.dataset = self.dataset.rename_columns(mapping)
        return self
    
    def transform(self):
        #Override in subclass
        raise NotImplementedError
    
    def split(self):
        dataset = self.dataset.train_test_split(
            test_size = self.config['data']['test_size'],
            seed = self.config['data']['seed'],
            shuffle = self.config['data']['shuffle']
        )

        return dataset
    
    def save(self, dataset, path):
        os.makedirs(path, exist_ok=True)
        dataset.save_to_disk(path)
        print(f"Dataset saved to {path}")

    def load_cached(self, path):
        return load_from_disk(path)

    def build(self):
        processed_dir = self.config['data']['processed_dir']
        
        if os.path.exists(processed_dir):
            print(f"Loading cached dataset from {processed_dir}")
            return self.load_cached(processed_dir)
        
        self.load()
        self.transform()
        dataset = self.split()
        self.save(dataset, processed_dir)
        return dataset