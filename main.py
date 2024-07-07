import argparse
import faiss
from PIL import Image
import json
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
import gradio as gr
import pandas as pd
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common.common import get_index_similarity
from common.constants import IMAGE_SHAPE
from utils.get_embedding import get_embedding


def search(image):
    global df
    global model
    global transform
    global device
    global index_inference

    embedding = get_embedding(model, image, transform, device)
    indexes = get_index_similarity(index_inference, embedding, 2)
    images = []
    for i in indexes:
        row = df.loc[i]
        image = Image.open(row['full_image_path'][1:]).convert('RGB')
        images.append(image)
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create faiss index')
    parser.add_argument('--db_files_path',
                        type=str,
                        default='../data/db/',
                        help='path to save faiss index and labels. Without name file.')
    parser.add_argument('--df_path', type=str, default='../data/dataset.csv', help='path to dataset (df)')
    parser.add_argument('--device', type=str, default='cpu', help='Use cpu or cuda device')

    args = parser.parse_args()
    device = args.device

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SHAPE, IMAGE_SHAPE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(f'{args.df_path}')
    df = df[df['part'] == 'train']
    df = df.reset_index(drop=True)

    # faiss
    faiss_index_file = f'{args.db_files_path}/database.index'
    faiss_label_file = f'{args.db_files_path}/train_labels.json'
    class_names_file = f'{args.db_files_path}/class_names.json'
    index_inference = faiss.read_index(faiss_index_file)

    with open(f'{faiss_label_file}') as f:
        labels_train = json.load(f)

    with open(f'{class_names_file}') as f:
        class_names = json.load(f)

    # model
    model = efficientnet_b0(pretrained=True)
    model.classifier = torch.nn.Identity()
    model.eval()

    # Создание интерфейса Gradio
    gr_interface = gr.Interface(fn=search,
                                inputs=gr.Image(type="pil"),
                                outputs=[gr.Image(type="pil"), gr.Image(type="pil")],
                                title="Image Similarity Search with FAISS")
    gr_interface.launch()
