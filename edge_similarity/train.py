from argparse import ArgumentParser
from pathlib import Path
from urllib.parse import urlparse

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

    datamodule = SymbolDataModule(args.dataset_path)
    logger = TensorBoardLogger(args.logdir, name="EdgeMetric")
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, gpus=1)

    with mlflow.start_run():
        mlflow.log_params({
            'lr': args.lr,
            'backbone': args.backbone,
            'epochs': args.epochs
        })

        model = EdgeMetric(args.backbone, args.lr)
        trainer.fit(model, datamodule=datamodule)

        metrics = trainer.validate(model, datamodule=datamodule)
        mlflow.log_metrics(metrics[0])

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.pytorch.log_model(model, "model", registered_model_name="EdgeMetric")
        else:
            mlflow.pytorch.log_model(model, "model")


if __name__ == '__main__':
    main()
