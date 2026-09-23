# EP39 — Multimodal Deep Learning: CLIP, BLIP & Vision-Language Models

> **Deep Learning Series · Episode 39 of 72 · Module 7 — Transformers & Attention**

ViT understands images. GPT understands text. **CLIP** makes them speak the same language.

Contrastive pretraining on **400 million** image-caption pairs → a single shared embedding space where an image of a cat and the sentence *"a photo of a cat"* become close vectors.

## What you will learn

- Multimodal learning & the alignment problem
- Contrastive pretraining + **InfoNCE** loss
- CLIP architecture (ViT image encoder + Transformer text encoder)
- Zero-shot image classification with natural language
- Full Hugging Face code walk-through
- BLIP (captioning + VQA)
- Multimodal LLMs (Flamingo, LLaVA, GPT-4V)
- Decision guide: when to use CLIP vs BLIP vs a multimodal LLM

## Key numbers

| Metric                        | Value                  |
|-------------------------------|------------------------|
| Training pairs                | 400 million            |
| Embedding dimension (ViT-B/32)| 512                    |
| Zero-shot ImageNet top-1      | **76.2%**              |
| Batch size used by OpenAI     | 32 768                 |
| Negatives per sample          | 32 767                 |

## Challenge

Run CLIP zero-shot classification on **10 of your own images**.

```bash
pip install transformers Pillow torch
