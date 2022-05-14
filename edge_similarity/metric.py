import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from effdet.efficientdet import _init_weight_alt, _init_weight
from effdet import get_efficientdet_config
from effdet.efficientdet import *

import utils


class EdgeMetric(pl.LightningModule):
    def __init__(self, backbone='d0', lr=0.001, agg='mean', precise_mask=False, unfreeze_backbone=False, reset_backbone=False):
        super().__init__()
        self.save_hyperparameters()

        config = get_efficientdet_config(f'tf_efficientdet_{backbone}')
        config.image_size = [64, 64]
        config.num_classes = 1
        config.min_level = 2
        config.num_levels = 6
        pretrained_backbone = not reset_backbone
        alternate_init = False

        self.config = config
        set_config_readonly(self.config)
        self.backbone = create_model(
            config.backbone_name, features_only=True,
            out_indices=self.config.backbone_indices or (1, 2, 3, 4),
            pretrained=pretrained_backbone, **config.backbone_args)
        if not unfreeze_backbone:
            self.backbone.requires_grad_(False)

        feature_info = get_feature_info(self.backbone)
        for fi in feature_info:
            fi['num_chs'] *= 2
        self.fpn = BiFpn(self.config, feature_info)
        self.class_net = nn.Sequential(
            SeparableConv2d(in_channels=config.fpn_channels, out_channels=config.fpn_channels, padding='same'),
            SeparableConv2d(in_channels=config.fpn_channels, out_channels=config.fpn_channels, padding='same'),
            SeparableConv2d(in_channels=config.fpn_channels, out_channels=config.num_classes, padding='same', bias=True,
                            norm_layer=None, act_layer=None),
            nn.Sigmoid()
        )

        for n, m in self.named_modules():
            if 'backbone' not in n:
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

    def aggregate(self, x):
        if self.hparams.agg == 'max':
            return torch.amax(x, dim=(-1, -2))
        elif self.hparams.agg == 'mean':
            return torch.mean(x, dim=(-1, -2))
        else:
            raise NotImplementedError(f'Aggregation "{self.hparams.agg}" not implemented')

    def forward(self, ref, tgt, return_heatmap=False):
        ref = self.backbone(ref)
        tgt = self.backbone(tgt)

        stacked = [torch.cat((r, t), dim=1) for r, t in zip(ref, tgt)]
        stacked = self.fpn(stacked)

        pred = self.class_net(stacked[0])

        if return_heatmap:
            return pred
        else:
            return self.aggregate(pred)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def loss_func(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)

    def log_heatmaps(self, src, pos, neg, semi, pos_res, neg_res, semi_res, mask, tag):
        fig, axes = plt.subplots(nrows=5, ncols=8)
        for i, ax in enumerate(axes):
            for a in ax:
                a.axis('off')

            ax[0].imshow(src[i].cpu().detach()[0], cmap='gray')
            ax[1].imshow(pos[i].cpu().detach()[0], cmap='gray')
            ax[2].imshow(pos_res[i].cpu().detach()[0], vmin=0, vmax=1)
            ax[3].imshow(neg[i].cpu().detach()[0], cmap='gray')
            ax[4].imshow(neg_res[i].cpu().detach()[0], vmin=0, vmax=1)
            ax[5].imshow(semi[i].cpu().detach()[0], cmap='gray')
            ax[6].imshow(semi_res[i].cpu().detach()[0], vmin=0, vmax=1)
            ax[7].imshow(mask[i].cpu().detach(), vmin=0, vmax=1)

        axes[0][0].set_title('Source')
        axes[0][1].set_title('Positive')
        axes[0][2].set_title('Pos Heatmap')
        axes[0][3].set_title('Negative')
        axes[0][4].set_title('Neg Heatmap')
        axes[0][5].set_title('Semi')
        axes[0][6].set_title('Semi Heatmap')
        axes[0][7].set_title('Mask')

        self.logger.experiment.add_figure(tag, fig, self.current_epoch)

    def base_step(self, batch, batch_idx, stage, log=True, figures=False):
        src, pos, neg, semi, mask = batch

        if figures and batch_idx == 0:
            with torch.no_grad():
                pos_res = self(src, pos, return_heatmap=True)
                neg_res = self(src, neg, return_heatmap=True)
        semi_res = self(src, semi, return_heatmap=True)

        # pos_mask = torch.zeros_like(mask)
        # if self.hparams.precise_mask:
        #     neg_mask = (torch.logical_or(neg[:, 0, ::4, ::4] != torch.amin(neg), ~neg[:, 0, ::4, ::4].eq(src[:, 0, ::4, ::4]))).type(mask.dtype).to(mask.device)
        # else:
        #     neg_mask = torch.ones_like(mask)

        # pos_loss = self.loss_func(pos_res.squeeze(1), pos_mask)
        # neg_loss = self.loss_func(neg_res.squeeze(1), neg_mask)
        semi_loss = self.loss_func(semi_res.squeeze(1), mask)
        # if self.hparams.precise_mask:
        #     loss = (pos_loss + neg_loss + semi_loss) / 3
        # else:
        loss = semi_loss

        if log:
            # self.log(f'{stage}/PosLoss', pos_loss)
            # self.log(f'{stage}/NegLoss', neg_loss)
            self.log(f'{stage}/SemiLoss', semi_loss, add_dataloader_idx=False)
            self.log(f'{stage}/Loss', loss, prog_bar=True, add_dataloader_idx=False)

            # self.log(f'{stage}/PosValue', pos_res.mean())
            # self.log(f'{stage}/NegValue', neg_res.mean())
            self.log(f'{stage}/SemiValue', semi_res.mean(), add_dataloader_idx=False)

        if figures and batch_idx == 0:
            self.log_heatmaps(src, pos, neg, semi, pos_res, neg_res, semi_res, mask, f'{stage}/Grid')

        return loss

    def training_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, 'Train')

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # self.base_step(batch, batch_idx, 'Val')

        if dataloader_idx == 0:
            self.base_step(batch, batch_idx, 'Val', figures=True)
        elif dataloader_idx == 1:
            name, ref, tgt = batch

            def _run(gt, image):
                return self(gt, image, True)

            res = utils.patch_metric(_run, ref, tgt, self.config.image_size[0], device=self.device)

            return name, res
        elif dataloader_idx == 2:
            self.base_step(batch, batch_idx, 'Train', log=False, figures=True)

    def validation_epoch_end(self, outputs):
        outputs = outputs[1]

        names = []
        heatmaps = []

        for name, heatmap in outputs:
            names.extend(name)
            heatmaps.append(heatmap)

        fig, axes = plt.subplots(nrows=2, ncols=2)
        for ax, name, heatmap in zip(axes.flatten(), names, heatmaps):
            ax.imshow(heatmap.cpu().detach(), vmin=0, vmax=1)
            ax.set_title(f'{name}: {heatmap.mean().item():.2f}')

        plt.tight_layout()
        self.logger.experiment.add_figure(f'Val/SR_res', fig, self.current_epoch)
