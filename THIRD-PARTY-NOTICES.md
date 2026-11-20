# Third-party material

This repository depends on, or instructs you to fetch, material that is not covered by [LICENSE](LICENSE)
or [LICENSE-DATA](LICENSE-DATA) and that the copyright holder of this repository cannot relicense. Each
item below keeps its own terms. Where those terms are more restrictive than this repository's licenses,
the more restrictive terms win for that material.

Status: refreshed 2026-11-20 to the current repository layout. Licence identifiers below were read from the
publishers' Hugging Face model cards or GitHub repositories on that date; they can change, so check the
source before relying on them.

## Code inside this repository

### SMOO (vendored subset) — `mmb/core.py`

`mmb/core.py` contains a vendored subset of the base abstractions of SMOO (`Criterion`,
`CriterionCollection`, `SUT`, `Manipulator`, `Optimizer`, the workflow base class). The StyleGAN-XL and
diffusion image backends still import further modules lazily from the external `smoo` package (optional
extras `stylegan` and `diffusion`); nothing in the default install needs it.

> **Unresolved:** the upstream repository (`oliverweissl/SMOO`) carried no license file when checked on
> 2026-11-20. Absent an express license the upstream author retains all rights, and this repository has no
> terms to pass on for that code. Treat the vendored subset and the optional `smoo` import as
> all-rights-reserved upstream code until a license is added there.

### Algorithms from PyPI packages

The search arms use the optimizers of `pymoo` (pinned in `pyproject.toml`): AGE-MOEA-II, NSGA-II and GA.
All Python dependencies are installed from PyPI, are not redistributed here, and carry their own licenses.

## Fetched at runtime, not redistributed

Model weights, checkpoints and datasets are downloaded when a pipeline runs; none are stored in this
repository, and the artifact bundle never contains them or images derived from them.

### Models

| Model | Used for | Licence as read on 2026-11-20 |
|---|---|---|
| `llava-hf/llava-v1.6-mistral-7b-hf` | SUT (main) | apache-2.0 (model card); built on LLaVA and Mistral, whose terms apply to the base models |
| `Qwen/Qwen3.5-4B` | SUT | apache-2.0 (model card) |
| `OpenGVLab/InternVL3_5-4B-HF` | SUT | apache-2.0 (model card) |
| `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen3.5-9B` | optional SUTs of the characterization experiments | Qwen model cards |
| `OpenVINO/Qwen2.5-VL-7B-Instruct-int{4,8}-ov`, `OpenVINO/llava-v1.6-mistral-7b-hf-int{4,8}-ov` | optional quantized SUTs (OpenVINO backend) | inherit the terms of the original models |
| `answerdotai/ModernBERT-large` | masked-LM synonym operator | apache-2.0 (model card) |
| `sentence-transformers/all-MiniLM-L6-v2` | text-embedding distance; validity proxy | apache-2.0 (model card) |
| `openai/clip-vit-base-patch32` | validity proxy | model card has no license field; the upstream `openai/CLIP` repository is MIT |
| `facebook/dinov2-small` | validity proxy | apache-2.0 (model card) |
| LPIPS (`lpips` package, AlexNet) | validity proxy | package repository: BSD-2-Clause; the AlexNet backbone weights come from torchvision |
| VQGAN f8-16384 (`latent-diffusion/vq-f8.zip`, downloaded from the `ommer-lab.com` host named in `mmb/manipulator/image/loading.py`) | image operator | checkpoint terms are set by its publisher; the `CompVis/latent-diffusion` code repository is MIT |
| `thomwolf/vqgan_imagenet_f16_1024`, `dalle-mini/vqgan_imagenet_f16_16384` | optional VQGAN variants | model cards |
| StyleGAN-XL `imagenet256.pkl` (optional backend) | image operator | **NVIDIA Source Code License-NC: noncommercial use only** |
| spaCy `en_core_web_sm` | tokenization | spaCy model terms |

The StyleGAN-XL checkpoint derives from NVIDIA StyleGAN3 code, released under a license restricting use to
non-commercial purposes. Any pipeline path that touches it is bound by that restriction independently of this
repository's licenses.

### Datasets

| Dataset | Used for | Terms |
|---|---|---|
| ImageNet-1k validation images (Hugging Face mirror `Tsomaros/Imagenet-1k_validation`) | Task 1 seeds | ImageNet / ILSVRC terms of access (noncommercial research); the mirror's card carries no license field. Photographs and images derived from them are not redistributed; cell lists carry identifiers only |
| POPE (`lmms-lab/POPE`), built on COCO val2014 images | Task 2 seeds | the dataset card carries no license field; COCO images remain under their Flickr terms of use. Not redistributed; cell lists carry identifiers only |

## Reporting a problem

If you hold rights in any material listed here and believe it is inappropriately included, contact
Ruben.Kaiser@tum.de and it will be removed.
