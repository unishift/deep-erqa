import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.widgets import Button, Slider
from torchvision.transforms.functional import to_pil_image


def get_padding(side, patch_size, stride):
    subd = side - patch_size
    return int(np.ceil(subd / stride)) * stride - subd


def smart_unfold(image, patch_size, stride=None):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    if stride is None:
        stride = patch_size

    *_, h, w = image.shape
    padding = (
        get_padding(h, patch_size, stride) // 2,
        get_padding(w, patch_size, stride) // 2,
    )
    image = F.unfold(image, patch_size, stride=stride, padding=padding)
    image = image.view(1, 3, patch_size, patch_size, -1)

    return image


def get_fold_divisor(image: torch.tensor, patch_size, stride, padding):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    # Remove redundant samples and channels
    image = image[:1, :1]

    input_ones = torch.ones(image.shape, dtype=image.dtype)
    divisor = F.fold(F.unfold(input_ones, patch_size, stride=stride, padding=padding), image.shape[-2:], patch_size,
                     stride=stride, padding=padding)

    return divisor


def probs_fold(probs, output_size, patch_size, stride, device='cpu'):
    probs = probs.view(probs.shape[0], -1, probs.shape[-1])

    h, w = output_size
    padding = (
        get_padding(h, patch_size, stride) // 2,
        get_padding(w, patch_size, stride) // 2,
    )
    probs_folded = F.fold(probs, output_size, patch_size, stride=stride, padding=padding)

    return probs_folded / get_fold_divisor(probs_folded, patch_size, stride, padding).to(device)


def patch_metric(func, gt, sample, patch_size, stride=None, device='cpu'):
    if stride is None:
        stride = patch_size

    gt_blocks = smart_unfold(gt, patch_size, stride)
    sample_blocks = smart_unfold(sample, patch_size, stride)

    gt_blocks = torch.movedim(gt_blocks, -1, 1)
    sample_blocks = torch.movedim(sample_blocks, -1, 1)

    batch_size, n_blocks, *shape = gt_blocks.shape

    gt_blocks = gt_blocks.view(batch_size * n_blocks, *shape)
    sample_blocks = sample_blocks.view(batch_size * n_blocks, *shape)

    probs = func(gt_blocks, sample_blocks)
    multiplier = sample_blocks.shape[-1] // probs.shape[-1]

    probs = probs.view(batch_size, n_blocks, -1)

    new_shape = (gt.shape[-2] // multiplier, gt.shape[-1] // multiplier)
    probs = torch.movedim(probs, 1, -1)
    probs = probs_fold(probs, new_shape, patch_size // multiplier, stride=stride // multiplier, device=device)
    probs = probs[:, 0].squeeze(0)

    return probs


class HeatmapState:
    def __init__(self, gt, sample, heatmap, ax=None):
        self.gt = gt
        self.sample = sample
        self.heatmap = heatmap

        self.show_sample = True
        if ax is None:
            self.fig, self.axes = plt.subplots(figsize=(7, 5))
        else:
            self.fig, self.axes = None, ax

        self.plot()

        axfreq = plt.axes([0.1, 0.03, 0.65, 0.03])
        self.freq_slider = Slider(
            ax=axfreq,
            label='Alpha',
            valmin=0.0,
            valmax=1.0,
            valinit=0.75,
        )
        self.freq_slider.on_changed(self.update)

        axbut = plt.axes([0.83, 0.03, 0.15, 0.05])
        self.button = Button(
            ax=axbut,
            label='Toggle',
        )
        self.button.on_clicked(self.toggle)

        plt.colorbar(self.bar, ax=self.axes)

    def plot_image(self, image, **kwargs):
        return self.axes.imshow(to_pil_image(image / 2 + 0.5), **kwargs)

    def plot_heatmap(self):
        self.bar = self.axes.imshow(self.heatmap, interpolation='bilinear', alpha=0.9, cmap='Reds')

    def plot(self):
        self.gt_bar = self.plot_image(self.gt, alpha=0)
        self.sample_bar = self.plot_image(self.sample)
        self.plot_heatmap()

    def update(self, event):
        self.bar.set(alpha=event)
        plt.draw()

    def toggle(self, event):
        self.show_sample = not self.show_sample
        if self.show_sample:
            self.gt_bar.set(alpha=0)
            self.sample_bar.set(alpha=1)
        else:
            self.gt_bar.set(alpha=1)
            self.sample_bar.set(alpha=0)
        plt.draw()
