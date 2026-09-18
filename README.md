# MM-RQRec: Multimodal Residual Quantization for Personalized Recommendation

**Tingwang Chen, Enhao Zhong, Meng Cai, and Yihao Zhao**

**Accepted for an oral presentation at the RetailVision Workshop @ ECCV 2026.**

[Paper (PDF)](https://openreview.net/pdf?id=GOLl4ZUvfY) | [OpenReview](https://openreview.net/forum?id=GOLl4ZUvfY) | [NineRec Codebase](https://github.com/westlake-repl/NineRec)

## Overview

MM-RQRec is a multimodal sequential recommendation framework that combines textual and visual item features with parameter-efficient encoder adaptation and residual quantization.

Generic pretrained features may not capture the subtle distinctions needed for recommendation. MM-RQRec adapts text and image encoders with **LoRA**, enhances their fused representations with **multi-level residual quantization**, and models user interaction sequences with a **BERT4Rec-style backbone**.

The framework retains continuous multimodal features and uses quantized features as an auxiliary enhancement. It does not replace item representations with discrete semantic IDs or generate recommendations token by token.

## Method

1. **Multimodal encoding:** A pretrained Chinese BERT encoder extracts textual features, and a pretrained Swin Transformer extracts visual features. The two representations are combined through late fusion.
2. **Parameter-efficient adaptation:** LoRA adapts both modality encoders to recommendation signals while keeping the original pretrained backbone weights frozen.
3. **Residual-quantized enhancement:** Multiple codebooks progressively quantize the residual of the fused representation. The resulting quantized feature is concatenated with the continuous feature, projected, and added back through a scaled residual connection.
4. **Sequential recommendation:** A BERT4Rec-style bidirectional Transformer models the enhanced item sequence. Training jointly optimizes a BPR-style recommendation loss and quantization regularization.

During inference, candidate item embeddings can be precomputed and cached. Recommendation scores are obtained by comparing the user sequence representation with candidate item embeddings.

## Codebase and Contributions

This implementation builds on [NineRec](https://github.com/westlake-repl/NineRec). We acknowledge the original authors for the underlying recommendation codebase and benchmark.

The MM-RQRec-specific work focuses on:

- Integrating LoRA-based adaptation into the text and image encoding pipeline.
- Enhancing continuous multimodal features with residual-quantized representations.
- Jointly optimizing recommendation and quantization objectives.
- Evaluating multimodal representations, encoder adaptation, and quantization strategies on NineRec-QB.

BERT4Rec, LoRA, and residual vector quantization are existing methods on which this framework builds.

## Results

The following results are reported in the paper on **NineRec-QB**. Higher is better for all metrics. These values are paper results, not an independent reproduction of this repository.

### Overall Performance

| Model | Recall@10 | NDCG@10 | AUC |
|---|---:|---:|---:|
| MoDVBPR | 0.010 | 0.004 | 0.656 |
| MoGRU4Rec | 0.055 | 0.028 | 0.824 |
| MoNextItNet | 0.049 | 0.027 | 0.759 |
| MoSASRec | 0.080 | 0.045 | 0.800 |
| MoBERT4Rec | 0.099 | 0.055 | 0.834 |
| **MM-RQRec** | **0.178** | **0.103** | **0.864** |

### Encoder Adaptation

| Adapted encoder | Recall@10 | NDCG@10 | AUC |
|---|---:|---:|---:|
| None | 0.099 | 0.055 | 0.834 |
| Text | 0.106 | 0.057 | 0.830 |
| Image | 0.127 | 0.072 | 0.839 |
| **Both** | **0.140** | **0.074** | **0.845** |

### Quantization Strategy

This comparison uses jointly adapted text and image encoders. The improvement from 0.099 to 0.178 Recall@10 in the overall table includes both encoder adaptation and quantization.

| Quantization strategy | Recall@10 | NDCG@10 | AUC |
|---|---:|---:|---:|
| Without quantization | 0.140 | 0.074 | 0.845 |
| VQ-VAE | 0.162 | 0.092 | 0.863 |
| **RQ-VAE** | **0.178** | **0.103** | **0.864** |

## Dataset

We use the **QB subset of NineRec**, containing user interaction sequences, item text, and cover images.

Please obtain the data from the [official NineRec repository](https://github.com/westlake-repl/NineRec#dataset) and follow its data usage instructions.

The upstream QB data includes:

| Resource | Contents |
|---|---|
| `QB_behaviour.tsv` | User interaction sequences |
| `QB_item.csv` | Item textual content |
| `QB_cover` | Item cover images |

The upstream repository provides image-to-LMDB preprocessing guidance. MM-RQRec-specific preprocessing options and local data paths remain to be documented.

## Reproducibility

### Experimental Configuration

Key settings reported in the paper:

| Setting | Value |
|---|---|
| Sequential backbone | BERT4Rec-style Transformer |
| Hidden dimension | 1024 |
| Transformer layers / attention heads | 2 / 4 |
| LoRA rank | 2 for both modality encoders |
| Residual codebooks | 2 |
| Entries per codebook | 128 |
| Residual enhancement scale | 0.1 |
| Quantization loss weight | 0.001 |
| Optimizer | AdamW |
| Recommendation / modality learning rate | 1e-4 / 1e-4 |
| Training / evaluation batch size | 32 / 1280 |
| Maximum epochs | 220 |
| Early-stopping metric | Validation NDCG@10 |
| LoRA training schedule | Updated for the first 9 epochs, then frozen |
| Experimental hardware | One NVIDIA A100 GPU |

### Installation, Training, and Evaluation

**Repository-specific reproduction instructions are not yet documented in this README.** The following information is needed to reproduce the reported experiments:

- Tested Python, PyTorch, CUDA, and package versions.
- Pretrained encoder identifiers and local checkpoint paths.
- Data preprocessing commands and configuration paths.
- Training commands, random seeds, and checkpoint selection details.
- Evaluation commands and the exact candidate-ranking protocol.

Refer to the paper for the experimental protocol and to NineRec for upstream documentation. Upstream commands alone should not be assumed to reproduce MM-RQRec.

## Limitations

- Experiments are conducted on NineRec-QB; generalization to other datasets requires further evaluation.
- Codebooks are not supervised using category labels. Quantization levels should not be interpreted as verified category hierarchies.
- The current framework uses fixed-weight late fusion.
- Item embedding extraction and periodic cache refreshes incur offline computation.

## Citation

If you use this work, please cite the paper. The following is a minimal citation for the OpenReview version:

```bibtex
@misc{chen2026mmrqrec,
  title = {{MM-RQRec}: Multimodal Residual Quantization for Personalized Recommendation},
  author = {Chen, Tingwang and Zhong, Enhao and Cai, Meng and Zhao, Yihao},
  year = {2026},
  howpublished = {OpenReview},
  url = {https://openreview.net/forum?id=GOLl4ZUvfY}
}
```

If you use the NineRec dataset or code, please also cite NineRec using the [citation provided by its authors](https://github.com/westlake-repl/NineRec#citation).

## Acknowledgments

We thank the authors of [NineRec](https://github.com/westlake-repl/NineRec) for making their benchmark and codebase available, and the authors of BERT4Rec, LoRA, and residual quantization for the methods underlying this work.

## Licensing

Third-party code, pretrained models, and datasets remain subject to their respective licenses and usage terms. This README does not grant a new license for upstream materials. Licensing terms for the MM-RQRec-specific additions remain to be specified.
