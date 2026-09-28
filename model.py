import logging
import os

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


logger = logging.getLogger(__name__)


class TextResidualAdapter(nn.Module):
    """Residual adapter used to build sketch-oriented text anchors."""

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
    """Load CLIP and return train/validation transforms.

    Weight resolution order:
    1. ASA_CLIP_WEIGHTS environment variable (local OpenCLIP checkpoint)
    2. --pretrained if it is an existing file path
    3. OpenCLIP pretrained tag, e.g. openai
    """
    log_func = logger.info if logger else print
    env_weights = os.environ.get("ASA_CLIP_WEIGHTS", "").strip()
    if env_weights:
        weight_source = env_weights
    elif pretrained and os.path.isfile(pretrained):
        weight_source = pretrained
    else:
        weight_source = pretrained
    log_func(f"Loading CLIP model: {model_name} ({weight_source})")

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

    model, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name, pretrained=weight_source
    )
    return model, safe_preprocess_train, preprocess_val


def encode_visual_with_prompt(visual, image, prompt):
    """ViT forward with extra visual tokens, matching CLIP-AT's concat-after-pos.

    CLIP-AT (Sketch_LVM) concatenates n_prompts tokens after the class+patch
    sequence and positional embedding, then runs ln_pre / transformer / CLS pool.
    """
    x = visual.conv1(image)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_token, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    patch_dropout = getattr(visual, "patch_dropout", None)
    if patch_dropout is not None:
        x = patch_dropout(x)
    if prompt is not None:
        x = torch.cat([x, prompt.to(dtype=x.dtype, device=x.device)], dim=1)
    x = visual.ln_pre(x)
    x = visual.transformer(x)
    pooled, _ = visual._pool(x)
    if visual.proj is not None:
        pooled = pooled @ visual.proj
    return pooled


class ClipAtEncoder(nn.Module):
    """CLIP-AT (Sain et al., CVPR 2023) under our protocol.

    Shared ViT, LayerNorm-only visual PEFT, and two sets of visual prompt
    tokens (sketch vs photo). Retrieval still uses encode_sketch / encode_photo.
    """

    def __init__(
        self,
        model_name="ViT-B-16",
        pretrained="openai",
        n_prompts=3,
        prompt_dim=768,
        logger=None,
    ):
        super().__init__()
        self.shared_model, self.preprocess_train, self.preprocess_val = get_model(
            model_name, pretrained, logger
        )
        for param in self.shared_model.parameters():
            param.requires_grad = False
        width = self.shared_model.visual.conv1.weight.shape[0]
        if prompt_dim != width:
            raise ValueError(
                f"CLIP-AT prompt_dim={prompt_dim} must match ViT width={width} "
                f"(ViT-B/16 is 768)."
            )
        self.n_prompts = n_prompts
        self.sk_prompt = nn.Parameter(torch.randn(n_prompts, prompt_dim) * 0.02)
        self.img_prompt = nn.Parameter(torch.randn(n_prompts, prompt_dim) * 0.02)

    def _expand_prompt(self, prompt, batch):
        return prompt.unsqueeze(0).expand(batch, -1, -1)

    def encode_sketch(self, sketch):
        prompt = self._expand_prompt(self.sk_prompt, sketch.size(0))
        return encode_visual_with_prompt(self.shared_model.visual, sketch, prompt)

    def encode_photo(self, photo):
        prompt = self._expand_prompt(self.img_prompt, photo.size(0))
        return encode_visual_with_prompt(self.shared_model.visual, photo, prompt)

    def encode_image(self, image):
        return self.encode_photo(image)

    def forward(self, sketch, photo, text_feat=None):
        sketch_feat = self.encode_sketch(sketch)
        photo_feat = self.encode_photo(photo)
        return sketch_feat, photo_feat, None


class MarlEncoder(nn.Module):
    """MARL (Lyou et al., WACV 2024) on our ViT-B/16 + LN protocol.

    Official code is a shared CLIP (B/32) plus mode embeddings that subtract
    modality from image/text and add them back for conversion. Retrieval uses
    sketch converted toward the photo mode vs raw photo gallery.

    freeze_all_but_bn exists in their repo but is never applied; we still only
    train visual LayerNorms (to match our CLIP-AT/single controls) plus the
    mode embedding and discriminator. CLIP text is frozen.
    """

    MODE_PHOTO = 0
    MODE_SKETCH = 1
    MODE_TEXT = 3

    def __init__(
        self,
        model_name="ViT-B-16",
        pretrained="openai",
        n_mode=4,
        clip_loss_ratio=1.0,
        semantic_loss_ratio=0.1,
        disc_loss_ratio=0.1,
        rec_loss_ratio=0.02,
        logger=None,
    ):
        super().__init__()
        self.shared_model, self.preprocess_train, self.preprocess_val = get_model(
            model_name, pretrained, logger
        )
        for param in self.shared_model.parameters():
            param.requires_grad = False
        embed_dim = self.shared_model.visual.output_dim
        self.n_mode = n_mode
        self.mode_embedding = nn.Embedding(n_mode, embed_dim)
        self.discriminator = nn.Linear(embed_dim, n_mode)
        self.clip_loss_ratio = clip_loss_ratio
        self.semantic_loss_ratio = semantic_loss_ratio
        self.disc_loss_ratio = disc_loss_ratio
        self.rec_loss_ratio = rec_loss_ratio
        self.clip_loss_fn = None

    def _visual(self, image):
        return self.shared_model.visual(image)

    def encode_photo(self, photo):
        return self._visual(photo)

    def encode_sketch(self, sketch):
        # Official retrieval: converted sketch → photo mode vs raw photos.
        z = self._visual(sketch)
        photo_m = self.mode_embedding.weight[self.MODE_PHOTO].to(dtype=z.dtype, device=z.device)
        sketch_m = self.mode_embedding.weight[self.MODE_SKETCH].to(dtype=z.dtype, device=z.device)
        return z - sketch_m + photo_m

    def encode_image(self, image):
        return self.encode_photo(image)

    def forward(self, sketch, photo, text_feat=None):
        return self.encode_sketch(sketch), self.encode_photo(photo), None

    def marl_loss(self, sketch, photo, text_feat, contrastive_fn):
        """Paired-batch version of MARL losses. text_feat is frozen CLIP text."""
        img = self._visual(photo)
        sk = self._visual(sketch)
        txt = F.normalize(text_feat.float(), p=2, dim=-1)
        img_n = F.normalize(img.float(), p=2, dim=-1)
        sk_n = F.normalize(sk.float(), p=2, dim=-1)

        clip_loss = 0.5 * (contrastive_fn(img, text_feat) + contrastive_fn(sk, text_feat))

        def _path_losses(image_n, image_mode):
            mode_img = self.mode_embedding.weight[image_mode]
            mode_txt = self.mode_embedding.weight[self.MODE_TEXT]
            e_s = F.normalize((txt + image_n) / 2.0, p=2, dim=-1)
            common_img = image_n - mode_img
            common_txt = txt - mode_txt
            ones = torch.ones(image_n.size(0), device=image_n.device)
            sem = F.cosine_embedding_loss(F.normalize(common_img, p=2, dim=-1), e_s, ones)
            sem = sem + F.cosine_embedding_loss(F.normalize(common_txt, p=2, dim=-1), e_s, ones)
            modes = torch.arange(self.n_mode, device=image_n.device)
            converted_img = F.normalize(
                common_img.unsqueeze(1) + self.mode_embedding(modes).unsqueeze(0), p=2, dim=-1
            )
            converted_txt = F.normalize(
                common_txt.unsqueeze(1) + self.mode_embedding(modes).unsqueeze(0), p=2, dim=-1
            )
            rec = F.cosine_embedding_loss(converted_txt[:, image_mode, :], image_n, ones)
            rec = rec + F.cosine_embedding_loss(converted_img[:, self.MODE_TEXT, :], txt, ones)
            labels = modes.unsqueeze(0).expand(image_n.size(0), -1).reshape(-1)
            ignore_img = labels == image_mode
            ignore_txt = labels == self.MODE_TEXT
            logits_img = self.discriminator(converted_img.reshape(-1, image_n.size(-1)).float())
            logits_txt = self.discriminator(converted_txt.reshape(-1, image_n.size(-1)).float())
            labels_img = labels.masked_fill(ignore_img, -1)
            labels_txt = labels.masked_fill(ignore_txt, -1)
            disc = 0.5 * (
                F.cross_entropy(logits_img, labels_img, ignore_index=-1)
                + F.cross_entropy(logits_txt, labels_txt, ignore_index=-1)
            )
            return sem, rec, disc

        sem_p, rec_p, disc_p = _path_losses(img_n, self.MODE_PHOTO)
        sem_s, rec_s, disc_s = _path_losses(sk_n, self.MODE_SKETCH)
        sem = 0.5 * (sem_p + sem_s)
        rec = 0.5 * (rec_p + rec_s)
        disc = 0.5 * (disc_p + disc_s)

        denom = (
            self.clip_loss_ratio
            + self.semantic_loss_ratio
            + self.disc_loss_ratio
            + self.rec_loss_ratio
        )
        loss = (
            self.clip_loss_ratio * clip_loss
            + self.semantic_loss_ratio * sem
            + self.disc_loss_ratio * disc
            + self.rec_loss_ratio * rec
        ) / denom
        return loss, {
            "clip": float(clip_loss.detach()),
            "semantic": float(sem.detach()),
            "disc": float(disc.detach()),
            "rec": float(rec.detach()),
        }


class ClipFitSingleEncoder(nn.Module):
    """Single-branch shared visual encoder for the single-vs-dual ablation."""

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
    """Dual visual encoders plus an optional residual text adapter."""

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
    """Freeze everything except selected visual LayerNorms and TextAdapter."""
    log_func = (lambda *args, **kwargs: None)
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

    if hasattr(model, "sk_prompt") and hasattr(model, "img_prompt"):
        model.sk_prompt.requires_grad = True
        model.img_prompt.requires_grad = True
        trainable.extend([model.sk_prompt, model.img_prompt])
        log_func("Unfrozen CLIP-AT visual prompt tokens.")

    if hasattr(model, "mode_embedding"):
        for param in model.mode_embedding.parameters():
            param.requires_grad = True
            trainable.append(param)
        log_func("Unfrozen MARL mode embeddings.")
    if hasattr(model, "discriminator"):
        for param in model.discriminator.parameters():
            param.requires_grad = True
            trainable.append(param)
        log_func("Unfrozen MARL discriminator.")

    if getattr(model, "use_text_adapter", False) and hasattr(model, "text_adapter"):
        for param in model.text_adapter.parameters():
            param.requires_grad = True
            trainable.append(param)
        log_func("Unfrozen TextResidualAdapter parameters.")

    log_func(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
    return trainable
