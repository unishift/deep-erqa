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
    parser.add_argument('--unmask-zeros', action='store_true')
    parser.add_argument('--precise-mask', action='store_true')
    parser.add_argument('--unfreeze-backbone', action='store_true')
    parser.add_argument('--reset-backbone', action='store_true')

    parser.add_argument('--logdir', default='/home/experiments/tb_logdir')
    parser.add_argument('--exp-name', default=None)
    parser.add_argument('--ckpt-path', type=Path, default=None)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('dataset_path', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.pytorch.autolog()

    mlflow.log_params({
        'backbone': args.backbone,
        'agg': args.agg,
        'canny': args.canny,
        'unmask_zeros': args.unmask_zeros,
        'precise_mask': args.precise_mask,
        'unfreeze_backbone': args.unfreeze_backbone,
        'reset_backbone': args.reset_backbone,
    })

    datamodule = SymbolDataModule(args.dataset_path, canny=args.canny, unmask_zeros=args.unmask_zeros)
    logger = TensorBoardLogger(args.logdir, name="EdgeMetric", version=args.exp_name)
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, gpus=[args.gpu])

    model = EdgeMetric(
        args.backbone, args.lr, agg=args.agg, precise_mask=args.precise_mask,
        unfreeze_backbone=args.unfreeze_backbone, reset_backbone=args.reset_backbone
    )
    if args.ckpt_path is not None:
        saved_model = mlflow.pytorch.load_model(args.ckpt_path)
        model.load_state_dict(saved_model.state_dict())
        del saved_model

    # trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
