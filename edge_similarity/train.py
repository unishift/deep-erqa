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
    parser.add_argument('--canny', action='store_true')

    parser.add_argument('--logdir', default='/home/experiments/tb_logdir')
    parser.add_argument('--exp-name', default=None)
    parser.add_argument('--ckpt-path', type=Path, default=None)

    parser.add_argument('dataset_path', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.pytorch.autolog()

    mlflow.log_params({
        'backbone': args.backbone,
        'agg': args.agg,
        'canny': args.canny
    })

    datamodule = SymbolDataModule(args.dataset_path, canny=args.canny)
    logger = TensorBoardLogger(args.logdir, name="EdgeMetric", version=args.exp_name)
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, gpus=1, auto_lr_find=True)

    if args.ckpt_path is None:
        model = EdgeMetric(args.backbone, args.lr, agg=args.agg)
    else:
        model = mlflow.pytorch.load_model(args.ckpt_path)
    trainer.fit(model, datamodule=datamodule)

    metrics = trainer.validate(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
