import argparse
import faiss
import json
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common.common import get_class_name
from common.common import get_label_to_class_dict
from common.constants import IMAGE_SHAPE
from utils.get_embedding import get_embedding


def main(args):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SHAPE, IMAGE_SHAPE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(f'{args.df_path}')
    df = df[df['part'] == 'test']

    # faiss
    faiss_index_file = f'{args.db_files_path}/database.index'
    faiss_label_file = f'{args.db_files_path}/train_labels.json'
    class_names_file = f'{args.db_files_path}/class_names.json'
    index_inference = faiss.read_index(faiss_index_file)

    with open(f'{faiss_label_file}') as f:
        labels_train = json.load(f)

    with open(f'{class_names_file}') as f:
        class_names = json.load(f)

    labels_to_class = get_label_to_class_dict(class_names)

    # model
    model = efficientnet_b0(pretrained=True)
    model.classifier = torch.nn.Identity()
    model.eval()

    # embedding
    row = df.sample(1).iloc[0]
    img = Image.open(row['full_image_path']).convert('RGB')
    embedding = get_embedding(model, img, transform, args.device)
    class_name = get_class_name(index_inference, embedding, labels_train, k=2)
    print(f'Predicted thing {labels_to_class[class_name]}\nTrue thing {row["label"]}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create faiss index')
    parser.add_argument('--db_files_path',
                        type=str,
                        default='../data/db/',
                        help='path to save faiss index and labels. Without name file.')
    parser.add_argument('--df_path', type=str, default='../data/dataset.csv', help='path to dataset (df)')
    parser.add_argument('--device', type=str, default='cpu', help='Use cpu or cuda device')

    args = parser.parse_args()
    main(args)
