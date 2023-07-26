## BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
This is the official implementation of BLIP-2 [paper](https://arxiv.org/abs/2301.12597), a generic and efficient pre-training strategy that easily harvests development of pretrained vision models and large language models (LLMs) for vision-language pretraining. BLIP-2 beats Flamingo on zero-shot VQAv2 (**65.0** vs **56.3**), establishing new state-of-the-art on zero-shot captioning (on NoCaps **121.6** CIDEr score vs previous best **113.2**). Equipped with powerful LLMs (e.g. OPT, FlanT5), BLIP-2 also unlocks the new **zero-shot instructed vision-to-language generation** capabilities for various interesting applications!

<img src="blip2_illustration.png" width="500">

### Install:
```
pip install https://github.com/m-bain/LAVIS/archive/main.zip
```


# download example image
`wget https://raw.githubusercontent.com/m-bain/LAVIS/main/docs/_static/merlion.png`

### Raw image feature extraction (EVA-VIT-G)

```python
import torch
from PIL import Image

from lavis.models import load_model_and_preprocess

raw_image = Image.open("merlion.png").convert("RGB")
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
sample = {"image": image}
features_image = model.extract_features(sample, mode="image_raw")
print(features_image.image_embeds.shape)
```

### Raw image feature extraction (CLIP-VIT-L)

```python
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
sample = {"image": image}
features_image = model.extract_features(sample, mode="image_raw")
print(features_image.image_embeds.shape)
```
