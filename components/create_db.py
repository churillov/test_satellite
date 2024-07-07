"""Create faiss database"""
import argparse
import faiss
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common.common import get_class_to_label
from common.constants import IMAGE_SHAPE
from utils.custom_dataset import FabulaDataset
from utils.get_embedding import prepare_data_features


def main(args):
    device = args.device
    index_path = f'{args.save_db_files_path}/database.index'
    labels_path = f'{args.save_db_files_path}/train_labels.json'
    class_names_path = f'{args.save_db_files_path}/class_names.json'

    df = pd.read_csv(f'{args.df_path}')
    df = df[df['part'] == 'train']

    dataset = FabulaDataset(df=df,
                            transform=transforms.Compose([
                                transforms.Resize((IMAGE_SHAPE, IMAGE_SHAPE)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ]))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    model = efficientnet_b0(pretrained=True)
    model.classifier = torch.nn.Identity()

    embeddings, labels = prepare_data_features(model, dataloader, device)
    print(embeddings.shape, labels.shape, df.shape, '\n')

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    labels_index = {}
    for ind, i in enumerate(labels):
        labels_index[f'{ind}'] = f'{i}'

    with open(f'{labels_path}', 'w') as file:
        json.dump(labels_index, file)

    class_to_label = get_class_to_label(df['label'].unique())
    with open(f'{class_names_path}', 'w') as file:
        json.dump(class_to_label, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create faiss index')
    parser.add_argument('--save_db_files_path',
                        type=str,
                        default='../data/db/',
                        help='path to save faiss index and labels. Without name file.')
    parser.add_argument('--df_path', type=str, default='../data/dataset.csv', help='path to dataset (df)')
    parser.add_argument('--device', type=str, default='cpu', help='Use cpu or cuda device')

    args = parser.parse_args()
    main(args)
