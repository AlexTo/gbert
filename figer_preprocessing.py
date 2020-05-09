import argparse
import pickle

from sklearn.preprocessing import MultiLabelBinarizer

from src.utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/FIGER", required=False)
    parser.add_argument("--output_dir", type=str, default="outputs/FIGER", required=False)
    return parser.parse_args()


def process_embeddings(input_dir, output_dir):
    if os.path.exists(f'{output_dir}/embeddings.bin') and os.path.exists(
            f'{output_dir}/ent_to_ind.bin') and os.path.exists(f'{output_dir}/ind_to_ent.bin'):
        with open(f'{output_dir}/embeddings.bin', 'rb') as f:
            embeddings = pickle.load(f)
        with open(f'{output_dir}/ent_to_ind.bin', 'rb') as f:
            ent_to_ind = pickle.load(f)
        with open(f'{output_dir}/ind_to_ent.bin', 'rb') as f:
            ind_to_ent = pickle.load(f)
    else:
        ent_to_ind = {}
        ind_to_ent = {}
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
                ind_to_ent[i] = entity
                split.pop(0)
                embeddings.append(np.array(split).astype(np.float))
                i = i + 1
        embeddings = np.array(embeddings)
        with open_file(f'{output_dir}', 'embeddings.bin', 'wb') as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open_file(f'{output_dir}', 'ent_to_ind.bin', 'wb') as f:
            pickle.dump(ent_to_ind, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open_file(f'{output_dir}', 'ind_to_ent.bin', 'wb') as f:
            pickle.dump(ind_to_ent, f, protocol=pickle.HIGHEST_PROTOCOL)
    return ent_to_ind, ind_to_ent, embeddings


def process_types(input_dir, output_dir):
    if os.path.exists(f'{output_dir}/type_to_ind.bin') and os.path.exists(f'{output_dir}/ind_to_type.bin'):
        with open(f'{output_dir}/type_to_ind.bin', 'rb') as f:
            type_to_ind = pickle.load(f)
        with open(f'{output_dir}/ind_to_type.bin', 'rb') as f:
            ind_to_type = pickle.load(f)
    else:
        type_to_ind = {}
        ind_to_type = {}
        type_freq = {}
        i = 0
        with open(f'{input_dir}/types', 'r') as f:
            for line in f:
                split = line.strip().split()
                type_ = split[0].strip()
                type_to_ind[type_] = i
                ind_to_type[i] = type_
                type_freq[type_] = int(split[2].strip())
                i = i + 1
        with open_file(f'{output_dir}', 'type_to_ind.bin', 'wb') as f:
            pickle.dump(type_to_ind, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open_file(f'{output_dir}', 'ind_to_type.bin', 'wb') as f:
            pickle.dump(ind_to_type, f, protocol=pickle.HIGHEST_PROTOCOL)
    return type_to_ind, ind_to_type


def process_data(input_dir, output_dir, num_types, ent_to_ind, type_to_ind, embeddings, ds='train'):
    features = []
    targets = []
    entities = []
    feature_dim = embeddings.shape[1]
    if os.path.exists(f"{output_dir}/{ds}_features.bin"):
        return

    with open(f'{input_dir}/E{ds}', 'r') as f:
        for line in f:
            split = line.strip().split()
            entity = split[0].strip()
            if entity in ent_to_ind:
                feature = embeddings[ent_to_ind[entity]]
                entities.append(ent_to_ind[entity])
            else:
                if ds != "train":
                    feature = np.zeros(feature_dim)
                    entities.append(-1)
                else:
                    continue
            if ds == 'test':
                targets.append([type_to_ind[t] for t in split[1:-1]])
            else:
                targets.append([type_to_ind[t] for t in split[1:]])
            features.append(feature)

    mlb = MultiLabelBinarizer(range(num_types))
    targets = mlb.fit_transform(targets)

    with open_file(f'{output_dir}', f'{ds}_entities.bin', 'wb') as f:
        pickle.dump(entities, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open_file(f'{output_dir}', f'{ds}_features.bin', 'wb') as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open_file(f'{output_dir}', f'{ds}_targets.bin', 'wb') as f:
        pickle.dump(targets, f, protocol=pickle.HIGHEST_PROTOCOL)

    if not os.path.exists(f"{output_dir}/type_adj.bin") and ds == "train":
        targets = torch.tensor(targets, dtype=torch.float)
        type_adj = batched_target_to_adj(targets)
        type_adj = batched_adj_to_freq(type_adj)
        type_adj[type_adj > 0] = 1
        type_adj = type_adj + torch.eye(num_types)
        type_adj = type_adj.numpy()
        with open_file(f'{output_dir}', 'type_adj.bin', 'wb') as f:
            pickle.dump(type_adj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    type_to_ind, ind_to_type = process_types(input_dir, output_dir)
    num_types = len(type_to_ind)

    ent_to_ind, ind_to_ent, embeddings = process_embeddings(input_dir, output_dir)

    process_data(input_dir, output_dir, num_types, ent_to_ind, type_to_ind, embeddings, ds="train")
    process_data(input_dir, output_dir, num_types, ent_to_ind, type_to_ind, embeddings, ds="dev")
    process_data(input_dir, output_dir, num_types, ent_to_ind, type_to_ind, embeddings, ds="test")


if __name__ == "__main__":
    main()
