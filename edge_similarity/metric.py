import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from effdet.efficientdet import _init_weight_alt, _init_weight
from effdet import get_efficientdet_config
from effdet.efficientdet import *


class EdgeMetric(pl.LightningModule):
    def __init__(self, backbone='d0', lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        config = get_efficientdet_config(f'tf_efficientdet_{backbone}')
        config.image_size = [64, 64]
        config.num_classes = 1
        pretrained_backbone = True
        alternate_init = False

        self.config = config
        set_config_readonly(self.config)
        self.backbone = create_model(
            config.backbone_name, features_only=True,
            out_indices=self.config.backbone_indices or (2, 3, 4),
            pretrained=pretrained_backbone, **config.backbone_args)
        feature_info = get_feature_info(self.backbone)
        for fi in feature_info:
            fi['num_chs'] *= 2
        self.fpn = BiFpn(self.config, feature_info)
        self.class_net = nn.Sequential(
            SeparableConv2d(in_channels=config.fpn_channels, out_channels=config.num_classes, padding='same', bias=True, norm_layer=None, act_layer=None),
        )

        for n, m in self.named_modules():
            if 'backbone' not in n:
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

    def forward(self, ref, tgt, return_heatmap=False):
        ref = self.backbone(ref)
        tgt = self.backbone(tgt)

        stacked = [torch.cat((r, t), dim=1) for r, t in zip(ref, tgt)]
        stacked = self.fpn(stacked)

        pred = self.class_net(stacked[0])

        if return_heatmap:
            return pred
        else:
            return torch.amax(pred, dim=(-1, -2))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        src, pos, neg = batch

        pos_res = self(src, pos)
        neg_res = self(src, neg)

        pos_labels = torch.zeros(len(src), dtype=src.dtype, device=src.device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_res.squeeze(1), pos_labels)
        neg_loss = F.binary_cross_entropy_with_logits(neg_res.squeeze(1), 1 - pos_labels)
        loss = (pos_loss + neg_loss) / 2

        self.log('Train/PosLoss', pos_loss)
        self.log('Train/NegLoss', neg_loss)
        self.log('Train/Loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        src, pos, neg = batch

        pos_res = self(src, pos, return_heatmap=True)
        neg_res = self(src, neg, return_heatmap=True)

        pos_value = torch.amax(pos_res, dim=(-1, -2))
        neg_value = torch.amax(neg_res, dim=(-1, -2))

        pos_labels = torch.zeros(len(src), dtype=src.dtype, device=src.device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_value.squeeze(1), pos_labels)
        neg_loss = F.binary_cross_entropy_with_logits(neg_value.squeeze(1), 1 - pos_labels)
        loss = (pos_loss + neg_loss) / 2

        self.log('Val/PosLoss', pos_loss)
        self.log('Val/NegLoss', neg_loss)
        self.log('Val/Loss', loss)

        fig, axes = plt.subplots(nrows=5, ncols=5)
        for i, ax in enumerate(axes):
            ax[0].imshow(src[i].cpu().detach()[0])
            ax[1].imshow(pos[i].cpu().detach()[0])
            ax[2].imshow(pos_res[i].cpu().detach()[0])
            ax[3].imshow(neg[i].cpu().detach()[0])
            ax[4].imshow(neg_res[i].cpu().detach()[0])


        # if dataloader_idx == 0:
        #     self.training_step(batch, batch_idx)
        # elif dataloader_idx == 1:
        #     name, ref, tgt = batch
        #
        #     res = self(ref, tgt)
        #
        #     return name, res

    # def validation_epoch_end(self, outputs):
    #     outputs = outputs[0]
    #
    #     names = []
    #     heatmaps = []
    #
    #     for name, heatmap in outputs:
    #         names.extend(name)
    #         heatmaps.extend(heatmap)
    #
    #     for name, heatmap in zip(names, heatmaps):
    #         self.logger.experiment.add_image(f'Val/{name}', heatmap, self.current_epoch, dataformats='CHW')
