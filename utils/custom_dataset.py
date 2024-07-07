"""Create custom dataset"""
from PIL import Image
from torch.utils.data import Dataset

from common.common import get_class_to_label


class FabulaDataset(Dataset):

    def __init__(self, df, transform=False):
        self.df = df
        self.transform = transform
        classes_name = self.df['label'].unique()
        self.class_to_label = get_class_to_label(classes_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['full_image_path']
        label_name = self.df.iloc[idx]['label']
        label = self.class_to_label[label_name]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
