import numpy as np
from skimage.segmentation import clear_border
from skimage.morphology import closing
from skimage.measure import label
from scipy.spatial.distance import cdist

def get_pockets_segmentation(density, threshold=0.5, min_size=50, scale=0.5):
    voxel_size = (1 / scale) ** 3
    bw = closing((density[0] > threshold).any(axis=0))
    cleared = clear_border(bw)
    label_image, num_labels = label(cleared, return_num=True)
    
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum() * voxel_size
        if pocket_size < min_size:
            label_image[np.where(pocket_idx)] = 0
    
    binding_score = density[0][density[0] > threshold].mean()
    return label_image, binding_score

def save_pocket_mol(density, origin, step, mol, dist_cutoff=4.5, expand_residue=True):
    coords = np.array([a.coords for a in mol.atoms])
    atom2residue = np.array([a.residue.idx for a in mol.atoms])
    max_len = max(len(a.atoms) for a in mol.residues)
    residue2atom = np.array([a_list + [-1]*(max_len - len(a_list)) for a_list in [[a.idx - 1 for a in r.atoms] for r in mol.residues]])
    
    pockets, binding_score = get_pockets_segmentation(density)
    pocket_atoms = []
    
    for pocket_label in range(1, pockets.max() + 1):
        indices = np.argwhere(pockets == pocket_label).astype('float32')
        indices *= np.asarray(step)
        indices += np.asarray(origin)
        distance = cdist(coords, indices)
        close_atoms = np.where((distance < dist_cutoff).any(axis=1))[0]
        if len(close_atoms) == 0:
            continue
        if expand_residue:
            residue_ids = np.unique(atom2residue[close_atoms])
            close_atoms = np.concatenate(residue2atom[residue_ids])
        pocket_atoms.append([int(idx) for idx in close_atoms])

    pocket_mols = []
    for pocket in pocket_atoms:
        pocket_mol = mol.clone
        atoms_to_del = (set(range(len(pocket_mol.atoms))) - set(pocket))
        pocket_mol.OBMol.BeginModify()
        for aidx in sorted(atoms_to_del, reverse=True):
            atom = pocket_mol.OBMol.GetAtom(aidx + 1)
            pocket_mol.OBMol.DeleteAtom(atom)
        pocket_mol.OBMol.EndModify()
        pocket_mols.append(pocket_mol)

    return pocket_mols, binding_score
