from argparse import ArgumentParser
from pathlib import Path

import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import SymbolDataModule
from metric import EdgeMetric


def parse_args():
    parser = ArgumentParser('Trainer for EdgeMetric')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--backbone', default='d0')
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--logdir', default='/home/restoration-metric/tb_logdir')

    parser.add_argument('dataset_path', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.pytorch.autolog()

    datamodule = SymbolDataModule(args.dataset_path)
    logger = TensorBoardLogger(args.logdir, name="EdgeMetric")
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, gpus=1)

    model = EdgeMetric(args.backbone, args.lr)
    trainer.fit(model, datamodule=datamodule)

    metrics = trainer.validate(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
