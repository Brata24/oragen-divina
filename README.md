# Audio-Visual Occlusion-Robust Gender Recognition and Age Estimation Approach Based on Multi-Task Cross-Modal Attention

## Abstract
> Gender recognition and age estimation are essential tasks within soft biometric systems, where identifying these characteristics supports a wide range of applications. In real-world scenarios, challenges such as partial facial occlusion complicate these tasks by obscuring crucial voice and facial characteristics. These challenges highlight the importance of development of robust and efficient approaches for gender recognition and age estimation. In this study, we develop a novel audio-visual Occlusion-Robust Gender Recognition and Age Estimation (ORAGEN) approach. The proposed approach is based on intermediate features of unimodal transformer-based models and two Multi-Task Cross-Modal Attention (MTCMA) blocks, which predict gender, age, and protective mask type using voice and facial characteristics. We conduct detailed cross-corpus experiments on the TIMIT, aGender, CommonVoice, LAGENDA, IMDB-Clean, AFEW, VoxCeleb2, and BRAVE-MASKS corpora. The proposed unimodal models outperform State-of-the-Art approaches for gender recognition and age estimation. We investigate the impact of various protective mask types on the performance of audio-visual gender recognition and age estimation. The results show that the current large-scale data are still insufficient for a robust gender recognition and age estimation in partial facial occlusion conditions. On the Test subset of the VoxCeleb2 corpus, the proposed approach showed Unweighted Average Recall (UAR) of 99.51%, Mean Absolute Error (MAE) of 5.42, and UAR of 100% for gender recognition, age estimation, and protective mask type recognition, respectively, while on the Test subset of the BRAVE-MASKS corpus, it showed UAR=96.63%, MAE=7.52, and UAR=95.87%, for the same tasks. These results indicate that using data of people wearing protective masks, as well as including the protective mask type recognition task, yields performance gains on all tasks considered. ORAGEN can be integrated into the OCEAN-AI framework for optimizing HR processes, as well as into expert systems with practical applications in various domains including forensics, healthcare, and industrial safety.

## Project Structure
The `src/` directory contains the project source code. A key shared module is `src/common/`, which groups reusable components used across data processing, training, and evaluation.

```text
src/
├─ audio/                             — Audio pipeline: augmentation, datasets, models, and runnable scripts
│  ├─ augmentation/                   — Audio-specific augmentations
│  │  └─ wave_augmentation.py         — Waveform-level augmentation utilities (noise, stretch, etc.)
│  ├─ data/                           — Audio datasets and dataloading logic
│  │  └─ agender_dataset.py           — Audio-only dataset for Age/Gender (unimodal) tasks
│  ├─ models/                         — Audio model architectures and wrappers
│  │  ├─ __init__.py                  — Module exports for audio models
│  │  ├─ audio_models.py              — Core audio model definitions (encoders/heads)
│  │  └─ audio_sota_model.py          — Strong/SOTA-oriented audio model configuration/implementation
│  ├─ run_convert_audio.py            — Script to convert/prepare audio (e.g., resample, segment, format)
│  ├─ run_train_unimodal_agender.py   — Script to train an audio-only Age/Gender model
│  └─ run_voice_activity_detector.py  — Script for VAD (voice activity detection) preprocessing/inference
│
├─ common/                            — Shared utilities reused by audio/video/fusion modules
│  ├─ augmentation/                   — Generic augmentations (project-wide)
│  │  └─ identity_augmentation.py     — No-op augmentation (passes input unchanged)
│  ├─ data/                           — Shared data helpers and preprocessors
│  │  ├─ common.py                    — Common dataset/pipeline utilities and data structures
│  │  ├─ data_preprocessors.py        — Raw → tensor preprocessing (feature extraction/tokenization/padding)
│  │  └─ grouping.py                  — Grouping/bucketing logic for efficient batching/training
│  ├─ loss/                           — Loss functions and their composition
│  │  └─ loss.py                      — Loss implementations (multi-task loss), weighting
│  ├─ models/                         — Shared model building blocks and common layers
│  │  ├─ __init__.py                  — Module exports for common model utilities
│  │  └─ common.py                    — Layers/modules/wrappers/initialization helpers
│  ├─ net_trainer/                    — Training orchestration and loops
│  │  └─ multitask_net_trainer.py     — Multi-task trainer (multiple objectives/measures, task-wise reporting)
│  ├─ utils/                          — General-purpose helpers used across the project
│  │  ├─ accuracy.py                  — Accuracy and related measures computations
│  │  └─ common.py                    — Misc utilities (seeding, logging helpers, path/config helpers, save/load)
│  └─ visualization/                  — Visualization of training and results
│     └─ visualize.py                 — Plotting helpers (curves, qualitative outputs, reports)
│
├─ configs/                           — Project configuration (defaults, paths, hyperparameters)
│  └─ default_config.py               — Default training/data configuration template
│
├─ fusion/                            — Multimodal (audio+video) fusion: datasets, feature extraction, models, training scripts
│  ├─ augmentation/                   — Augmentations that operate on modalities or modality presence
│  │  └─ modality_augmentation.py     — Modality-level augmentation (drop/mask/perturb modalities, sync, etc.)
│  ├─ data/                           — Multimodal datasets and feature-based datasets
│  │  ├─ agender_multimodal_dataset.py            — Audio+video dataset for Age/Gender (raw or aligned inputs)
│  │  └─ agender_multimodal_features_dataset.py   — Dataset based on precomputed multimodal features/embeddings
│  ├─ features/                       — Feature extraction layer for multimodal pipelines
│  │  ├─ common.py                    — Shared feature utilities and feature container helpers
│  │  └─ feature_extractors.py        — Feature extractor definitions (audio/video encoders, embedding builders)
│  ├─ models/                         — Fusion models and wrappers
│  │  ├─ models_wrappers.py           — Wrappers around encoders/fusion blocks (load, freeze, multi-head, etc.)
│  │  └─ multimodal_models.py         — Multimodal fusion architectures (concat/attention/gating/etc.)
│  ├─ run_train_multimodal_agender.py — Script to train multimodal Age/Gender model
│  └─ run_train_multimodal_maskagender.py — Script to train multimodal model with modality masking strategy
│
└─ video/                             — Video pipeline: augmentation and video-specific models
   ├─ augmentation/                   — Video/image augmentations (spatial, color, crop, etc.)
   │  └─ image_augmentation.py        — Image/frame augmentation utilities for video training
   └─ models/                         — Video model architectures
      └─ video_models.py              — Core video model definitions (frame encoder, temporal pooling, heads)
