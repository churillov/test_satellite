"""Create df database"""
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def main(args):
    path_to_images = Path(f'{args.path_to_images}')
    path_to_csv = Path(f'{args.path_save_csv}/dataset.csv')

    df = pd.DataFrame(columns=['full_image_path', 'image_name', 'label', 'part'])
    for class_dir in path_to_images.iterdir():
        image_paths = list(class_dir.iterdir())
        X_train, X_test = train_test_split(image_paths, test_size=0.2, random_state=42)

        for image_path in X_train:
            df.loc[len(df.index)] = [image_path, image_path.name, class_dir.name, 'train']

        for image_path in X_test:
            df.loc[len(df.index)] = [image_path, image_path.name, class_dir.name, 'test']

    df.to_csv(path_or_buf=path_to_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create df database')
    parser.add_argument('--path_to_images',
                        type=str,
                        default='../data/dataset',
                        help='Path to images')
    parser.add_argument('--path_save_csv',
                        type=str,
                        default='../data/',
                        help='Path to save csv file. Without name file.')

    args = parser.parse_args()
    main(args)

