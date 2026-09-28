# CLIP checkpoint (not included)

Place a local OpenCLIP ViT-B/16 OpenAI checkpoint here only if the machine cannot download weights.

Suggested filename:

```text
pretrained/open_clip_vit_b16_openai.bin
```

Then:

```bash
export ASA_CLIP_WEIGHTS="$PWD/pretrained/open_clip_vit_b16_openai.bin"
```

Do not commit `.bin` / `.pth` files. `open_clip` can also load the `openai` tag directly when Hugging Face / OpenCLIP downloads are allowed.

==== loss.py：相对 23dbc24 无 diff，直接复制投稿版。SHA256 6dec7df73f50b578546ac51cda886abb06b237085e75a2376d9384575a502d8f
