from typing import Tuple
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm


@torch.no_grad()
def prepare_data_features(model, dataloader: DataLoader, device: str) -> Tuple[Tensor, Tensor]:
    """
    The function generates image embeddings.
    Args:
        model: Model.
        dataloader: DataLoader.
        device: Cuda or CPU.

    Returns:
        The function returns embeddings and labels.
    """
    model.eval()
    model.to(device)

    embeddings, labels = [], []
    for batch_images, batch_labels in tqdm(dataloader):
        batch_images = batch_images.to(device)
        with torch.no_grad():
            embeddings.append(model(batch_images).detach().cpu())
            labels.append(batch_labels)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    return embeddings, labels


@torch.no_grad()
def get_embedding(model, image: Image, transform: Compose, device: str = 'cpu') -> Tensor:
    """
    The function uses the model to obtain the image embedding.
    Args:
        model: Model.
        image: Image.
        transform: Transformations.
        device: Cuda or CPU.

    Returns:
        Return embedding.
    """
    model.eval()
    model.to(device)
    image = transform(image)[None, :, :, :]
    image = image.to(device)
    embedding = model(image).detach().cpu()
    return embedding
