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
    parser.add_argument('--agg', choices=('max', 'mean'), default='max')

    parser.add_argument('--logdir', default='/home/experiments/tb_logdir')

    parser.add_argument('dataset_path', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.pytorch.autolog()

    mlflow.log_params({
        'backbone': args.backbone,
    })

    datamodule = SymbolDataModule(args.dataset_path)
    logger = TensorBoardLogger(args.logdir, name="EdgeMetric")
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, gpus=1, auto_lr_find=True)

    model = EdgeMetric(args.backbone, args.lr, agg=args.agg)
    # trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)

    metrics = trainer.validate(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
