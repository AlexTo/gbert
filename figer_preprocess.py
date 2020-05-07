import pickle

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from src.utils import *

num_types = 102
input_dir = 'data/FIGER'
output_dir = 'outputs/FIGER'
mlb = MultiLabelBinarizer(range(num_types))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_embeddings():
    if os.path.exists(f'{output_dir}/embeddings.bin') and os.path.exists(f'{output_dir}/ent_to_ind.bin'):
        with open(f'{output_dir}/embeddings.bin', 'rb') as f:
            embeddings = pickle.load(f)
        with open(f'{output_dir}/ent_to_ind.bin', 'rb') as f:
            ent_to_ind = pickle.load(f)
    else:
        ent_to_ind = {}
        i = 0
        embeddings = []
        with open(f'{input_dir}/embeddings.txt', 'r') as f:
            for line in f:
                split = line.strip().split()
                if i == 0:
                    vector_size = len(split) - 1
                if len(split) < vector_size + 1:
                    print(f'Skipping line: {line}')
                    continue
                entity = split[0].strip()
                ent_to_ind[entity] = i
                split.pop(0)
                embeddings.append(np.array(split).astype(np.float))
                i = i + 1
        embeddings = np.array(embeddings)
        with open_file(f'{output_dir}', 'embeddings.bin', 'wb') as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open_file(f'{output_dir}', 'ent_to_ind.bin', 'wb') as f:
            pickle.dump(ent_to_ind, f, protocol=pickle.HIGHEST_PROTOCOL)

    return ent_to_ind, embeddings


def load_types():
    type_to_ind = {}
    type_freq = {}
    i = 0
    with open(f'{input_dir}/types', 'r') as f:
        for line in f:
            split = line.strip().split()
            type_ = split[0].strip()
            type_to_ind[type_] = i
            type_freq[type_] = int(split[2].strip())
            i = i + 1
    return type_to_ind, type_freq


def load_data(ent_to_ind, type_to_ind, embeddings, ds_type='train'):
    features = []
    targets = []
    with open(f'{input_dir}/E{ds_type}', 'r') as f:
        for line in f:
            split = line.strip().split()
            entity = split[0].strip()
            if entity not in ent_to_ind:
                print(f"Skipping entity {entity}")
                continue
            feature = embeddings[ent_to_ind[entity]]
            targets.append([type_to_ind[t] for t in split[1:]])
            features.append(feature)

    targets = mlb.fit_transform(targets)

    return features, targets


def main():
    type_to_ind, type_freq = load_types()
    ent_to_ind, embeddings = load_embeddings()
    features, targets = load_data(ent_to_ind, type_to_ind, embeddings)
    targets = np.array(targets, np.int32)
    targets = sp.csc_matrix(targets)
    coo, val = target_to_sparse_adj(targets)
    i = 0


if __name__ == "__main__":
    main()
