import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import mlflow.pytorch
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import VSRbenchmark
from utils import save_image


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--gpu', type=int, default=None)

    parser.add_argument('model_path', type=Path)
    parser.add_argument('data_path', type=Path)
    parser.add_argument('save_path', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = VSRbenchmark(args.data_path)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8)

    model = mlflow.pytorch.load_model(str(args.model_path))
    trainer = pl.Trainer(gpus=[args.gpu])

    predictions = trainer.predict(model, dataloaders=dataloader)

    args.save_path.mkdir(exist_ok=True)

    probs_dict = defaultdict(lambda: [])
    for path, (prob, heatmap) in tqdm(list(zip(dataset.frames, predictions))):
        model_name = path.parent.name

        probs_dict[model_name].append(prob.item())

        vis_dir = args.save_path / 'vis' / model_name
        full_dir = args.save_path / 'full' / model_name

        vis_dir.mkdir(exist_ok=True, parents=True)
        full_dir.mkdir(exist_ok=True, parents=True)

        save_image(heatmap.squeeze(0).numpy(), vis_dir / path.name, vmin=0, vmax=1)
        torch.save(heatmap, full_dir / f'{path.stem}.pkl')

    with open(args.save_path / 'overall.json', 'w') as f:
        json.dump(probs_dict, f)


if __name__ == '__main__':
    main()
