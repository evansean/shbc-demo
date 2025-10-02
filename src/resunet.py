import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torchvision.transforms.functional import resize


class SegmentationModel(pl.LightningModule):
    def __init__(self, num_foregrounds):
        super().__init__()

        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            classes=num_foregrounds,
            in_channels=3
        )
        self.encoder_dropout = torch.nn.Dropout2d(p=0.5)
        self.decoder_dropout = torch.nn.Dropout2d(p=0.5)

    def _forward_encoder(self, x):
        model = self.model
        features = model.encoder(x)

        features[-1] = self.encoder_dropout(features[-1])

        return features

    def _forward_decoder(self, features):
        model = self.model
        decoder_output = model.decoder(*features)
        decoder_output = self.decoder_dropout(decoder_output)

        return model.segmentation_head(decoder_output)

    def forward(self, x):
        features = self._forward_encoder(x)
        logits = self._forward_decoder(features)

        return features, logits

    def get_inference_logits(self, x):
        return self(x)[-1]

    def inference(self, x, target_height, target_width):
        logits = self.get_inference_logits(x)
        y_hat = torch.sigmoid(logits)
        y_hat = resize(y_hat, (target_height, target_width))

        return y_hat
