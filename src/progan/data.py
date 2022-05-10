from torch.utils.data import DataLoader
from math import log2
import config
from src.data_factory.percolation import generate_percolation_data

def get_loader(image_size, dataset_size=5000):
    
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset, _ = generate_percolation_data(dataset_size=dataset_size, lattice_size=image_size, p_list=[0.5928], split=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset