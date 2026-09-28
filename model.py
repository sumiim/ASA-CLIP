import os

import open_clip
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class TextResidualAdapter(nn.Module):
    def __init__(self, embed_dim, bottleneck=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, embed_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, text_features):
        return F.normalize(text_features + self.net(text_features), p=2, dim=-1)


def get_model(model_name="ViT-B-16", pretrained="openai", logger=None):
    log_func = logger.info if logger else print
    log_func(f"Loading CLIP model: {model_name} ({pretrained})")
    local_weights_path = os.environ.get("OPENCLIP_WEIGHTS")

    safe_preprocess_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.9, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    pretrained_source = local_weights_path or pretrained
    try:
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_source
        )
    except Exception:
        if not local_weights_path:
            raise
        log_func(
            f"Could not load OPENCLIP_WEIGHTS={local_weights_path}; "
            f"falling back to pretrained='{pretrained}'."
        )
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

    return model, safe_preprocess_train, preprocess_val


class ClipFitSingleEncoder(nn.Module):
    def __init__(self, model_name="ViT-B-16", pretrained="openai", logger=None):
        super().__init__()
        self.shared_model, self.preprocess_train, self.preprocess_val = get_model(
            model_name, pretrained, logger
        )
        for param in self.shared_model.parameters():
            param.requires_grad = False

    def forward(self, sketch, photo, text_feat=None):
        sketch_feat = self.shared_model.visual(sketch)
        photo_feat = self.shared_model.visual(photo)
        text_feat = self.encode_text(text_feat) if text_feat is not None else None
        return sketch_feat, photo_feat, text_feat

    def encode_sketch(self, sketch):
        return self.shared_model.visual(sketch)

    def encode_photo(self, photo):
        return self.shared_model.visual(photo)

    def encode_image(self, image):
        return self.encode_photo(image)

    def encode_text(self, text_features):
        if text_features is None:
            return None
        return F.normalize(text_features, p=2, dim=-1)


class ClipFitDualEncoderWithTextAdapter(nn.Module):
    def __init__(
        self,
        model_name="ViT-B-16",
        pretrained="openai",
        use_text_adapter=True,
        text_bottleneck=256,
        logger=None,
        **kwargs,
    ):
        super().__init__()
        self.use_text_adapter = use_text_adapter
        self.sketch_model, self.preprocess_train, self.preprocess_val = get_model(
            model_name, pretrained, logger
        )
        self.image_model, _, _ = get_model(model_name, pretrained, logger)

        for param in self.sketch_model.parameters():
            param.requires_grad = False
        for param in self.image_model.parameters():
            param.requires_grad = False

        embed_dim = self.sketch_model.visual.output_dim
        if self.use_text_adapter:
            self.text_adapter = TextResidualAdapter(embed_dim, text_bottleneck)
            pass

        self.logit_scale = self.sketch_model.logit_scale

    def forward(self, sketch, photo, text_feat=None):
        sketch_feat = self.sketch_model.visual(sketch)
        photo_feat = self.image_model.visual(photo)
        adapted_text = self.encode_text(text_feat) if text_feat is not None else None
        return sketch_feat, photo_feat, adapted_text

    def encode_text(self, text_feat):
        if text_feat is None:
            return None
        text_feat = F.normalize(text_feat, p=2, dim=-1)
        if self.use_text_adapter:
            return self.text_adapter(text_feat)
        return text_feat

    def encode_sketch(self, sketch):
        return self.sketch_model.visual(sketch)

    def encode_photo(self, photo):
        return self.image_model.visual(photo)

    def encode_image(self, image):
        return self.encode_photo(image)


def configure_clipfit_parameters(model, logger=None, train_sketch=True, train_photo=True):
    log_func = logger.info if logger else print
    for param in model.parameters():
        param.requires_grad = False

    trainable = []

    def unfreeze_layernorms(module):
        for submodule in module.modules():
            if isinstance(submodule, nn.LayerNorm):
                submodule.weight.requires_grad = True
                submodule.bias.requires_grad = True
                trainable.extend([submodule.weight, submodule.bias])

    if hasattr(model, "sketch_model"):
        if train_sketch:
            unfreeze_layernorms(model.sketch_model.visual)
            log_func("Unfrozen sketch encoder LayerNorm affine parameters.")
        if train_photo:
            unfreeze_layernorms(model.image_model.visual)
            log_func("Unfrozen photo encoder LayerNorm affine parameters.")
    elif hasattr(model, "shared_model"):
        if train_sketch or train_photo:
            unfreeze_layernorms(model.shared_model.visual)
            log_func("Unfrozen shared encoder LayerNorm affine parameters.")

    if getattr(model, "use_text_adapter", False) and hasattr(model, "text_adapter"):
        for param in model.text_adapter.parameters():
            param.requires_grad = True
            trainable.append(param)
        log_func("Unfrozen TextResidualAdapter parameters.")

    log_func(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
    return trainable
