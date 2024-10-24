from proteindata import proteinDataset_predict
from torch.utils.data import DataLoader
from openbabel import pybel

def prepare_data(input_file, file_format):
    val_dataset = proteinDataset_predict(data_path=input_file, file_format=file_format)
    val_Dataloader = DataLoader(val_dataset, batch_size=1, num_workers=1)
    return val_Dataloader

def process_protein_data(data, file_format):
    input, origin, step, name, mol = data
    mol = next(pybel.readfile(file_format, mol[0]))
    return input, origin, step, name, mol
