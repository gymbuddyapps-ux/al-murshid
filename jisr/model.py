"""The sign model: a small 1D convolutional network (1D-CNN).

Input:  one clip as 32 frames x 140 numbers (see docs/features.md).
Output: one score per class ("logits"). A higher score means the model thinks
        that class is more likely. Softmax turns the scores into probabilities.

A 1D convolution slides a small window along the time axis of the clip and looks
for short movement patterns (3 frames at a time). Stacking three of them lets the
model see longer movements. At the end, the patterns found anywhere in the clip
are summarized (average and maximum over time) and a final layer picks the class.
"""
import torch
from torch import nn

from jisr.features import FEATURE_SIZE, NUM_FRAMES


class SignCNN(nn.Module):
    def __init__(self, num_classes, hidden=(64, 96, 128), dropout=0.3):
        super().__init__()
        h1, h2, h3 = hidden
        self.conv = nn.Sequential(
            nn.Conv1d(FEATURE_SIZE, h1, kernel_size=3, padding=1), nn.BatchNorm1d(h1), nn.ReLU(),
            nn.Conv1d(h1, h2, kernel_size=3, padding=1), nn.BatchNorm1d(h2), nn.ReLU(),
            nn.MaxPool1d(2),                                   # 32 frames -> 16
            nn.Conv1d(h2, h3, kernel_size=3, padding=1), nn.BatchNorm1d(h3), nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)                     # helps against overfitting
        self.classify = nn.Linear(2 * h3, num_classes)

    def forward(self, x):
        # x: (batch, 32 frames, 140 numbers). Conv1d wants (batch, 140, 32).
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Summarize over time: the average and the maximum of every pattern.
        x = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)
        return self.classify(self.dropout(x))


def export_onnx(model, path):
    """Save the model in the ONNX format, which the browser can run."""
    model.eval()
    example = torch.zeros(1, NUM_FRAMES, FEATURE_SIZE)
    torch.onnx.export(model, example, str(path), input_names=["features"],
                      output_names=["scores"], opset_version=17, dynamo=False)
