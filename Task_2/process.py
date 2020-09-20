import argparse
import json
import torch
from tqdm import tqdm
import torchvision
from torchvision import transforms
import shutil
import os


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_data(data_root):
    shutil.copytree(data_root, os.path.join(data_root, 'unknown'))

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ImageFolderWithPaths(data_root, data_transforms)
    data = torch.utils.data.DataLoader(dataset,
                                        batch_size=len(dataset),
                                        shuffle=False)
    return data


def predict(data, model):
    model.eval()
    predictions = {}
    classes = {0: 'female', 1: 'male'}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for inputs, labels, paths in tqdm(data):
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            preds = model(inputs)
            preds_class = preds.argmax(dim=1)
        for i, path in enumerate(paths):
            path = path.split('/')[-1]
            predictions[path] = classes[preds_class[i].item()]

    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help="data_root")
    args = parser.parse_args()
    data_root = args.folder

    model = torch.load('model_trained.pt')
    data = get_data(data_root)
    predictions = predict(data, model)

    with open('process_results.json', 'w') as f:
        json.dump(predictions, f, indent=4)
