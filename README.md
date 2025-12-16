## Disclaimer

This repository is **based on the official implementation** of  
**3D Gaussian Splatting for Real-Time Radiance Field Rendering**  
by Kerbl et al. (SIGGRAPH 2023).

All original copyrights belong to the authors of the original repository.
This project focuses on **empirical analysis and practical modifications**
to the training pipeline, aiming to improve boundary sharpness and background
stability in indoor scenes.

The original codebase can be found at:
https://github.com/graphdeco-inria/gaussian-splatting

# An Empirical Study on 3D Gaussian Splatting and NeRF for Indoor Scene Reconstruction

This repository documents a series of **systematic experiments conducted while learning, reproducing, and modifying 3D Gaussian Splatting (3DGS) and NeRF-based methods** for indoor scene reconstruction.

The focus of this project is **not to propose a new method**, but to:
- understand how existing methods behave in practice,
- identify failure cases in indoor environments,
- explore practical engineering-level modifications,
- and clarify the limitations of current representations.

This repository should be read as an **experimental and engineering study**, rather than a polished algorithmic contribution.

![assets/charts/customdata//comparison-0.png](assets/charts/customdata//comparison-0.png)

---

## 1. Scope of This Project

This project includes experiments on:

- Reproducing official 3DGS results
- Running 3DGS on simulated indoor scenes
- Running 3DGS on self-captured real-world datasets
- Data processing with COLMAP / SfM
- Multiple modifications to the original 3DGS pipeline
- Diffusion / residual-based refinement (as one experimental branch)
- Comparison with NeRF-based methods (Instant-NGP)

The goal is to **understand what works, what partially works, and what fundamentally fails** in indoor scenes.

---

## 2. Baseline Reproduction: Official 3DGS

We first reproduced the official 3D Gaussian Splatting pipeline using:
- the released implementation,
- provided example datasets,
- default training and rendering settings.

This step was necessary to:
- verify correct environment setup,
- understand baseline behavior,
- establish a reference point for all later experiments.

Baseline results matched the expected behavior reported by the authors.

---

## 3. Experiments on Custom Data

### 3.1 Simulated Indoor Scenes

We initially experimented with **simulated indoor environments**, including:
- simple geometric scenes,
- scenes with robots and basic indoor layouts.

Although simulation provides full control over geometry and camera poses, we observed:

- severe lack of visual features in texture-less regions,
- unstable or failed COLMAP reconstructions,
- unreliable camera pose estimation.

As a result, the simulation-based pipeline was **not suitable for SfM-based 3DGS**, and this direction was abandoned.

---

### 3.2 First Real-World Dataset (Failure Case)

We then captured a first real-world indoor dataset using a handheld camera.

Observed results:
- reconstruction was technically possible,
- but large background regions were extremely blurry,
- wall-floor and wall-object boundaries were unstable,
- results were visually unsatisfactory.

This dataset is kept as a **documented failure case** to analyze the causes of degradation.

---

### 3.3 Improved Real-World Dataset

Based on the failure analysis, we improved:
- camera trajectory and coverage,
- viewpoint diversity,
- capture consistency.

With the improved dataset:
- COLMAP reconstruction became stable,
- baseline 3DGS converged reliably,
- but background and boundary issues remained.

---

## 4. Data Processing Pipeline

We explicitly handled and inspected:
- COLMAP feature extraction and matching,
- camera pose alignment and scaling,
- image resolution and cropping,
- training/test split consistency.

This step emphasized that **3DGS is not an end-to-end black box**, and data quality strongly influences the final result.

---

## 5. Modifications to the Original 3DGS Pipeline

A series of engineering-level modifications were explored.

### 5.1 Opacity / Transparency Handling

We experimented with:
- modified opacity regularization,
- transparency-related loss terms.

Effect:
- background instability was partially reduced,
- but excessive regularization caused loss of detail.

---

### 5.2 Densification and Threshold Scheduling

We tested different:
- densification thresholds,
- pruning schedules,
- early vs late densification strategies.

Observations:
- aggressive densification improves edge sharpness,
- but introduces noise and shading artifacts,
- conservative settings improve cleanliness but blur boundaries.

---

### 5.3 Edge-Aware Background Treatment

We explicitly targeted:
- wall-floor intersections,
- wall-object boundaries,
- large planar background regions.

Results:
- geometric boundaries became sharper,
- but shading-like artifacts appeared,
- indicating a trade-off between structure and appearance.

---

## 6. Diffusion / Residual-Based Refinement (Experimental Branch)

As one experimental direction, we explored **diffusion-based and residual-based refinement**.

This includes:
- training a Detail Completion Model (DCM),
- learning residuals between 3DGS renders and ground truth,
- injecting the predicted residuals during training.

Important notes:
- this module operates **during training**, not as post-processing,
- it improves perceptual cleanliness and boundary stability,
- but does **not recover missing textures** in repetitive planar regions.

This branch is treated as an **experimental component**, not the core of the project.

---

## 7. Comparison with NeRF-Based Methods (Instant-NGP)

We compared 3DGS with Instant-NGP on the same datasets.

Key observations:
- NeRF-based methods better handle continuous textures,
- 3DGS excels at sharp boundaries and real-time rendering,
- repetitive planar textures remain challenging for both.

This highlights fundamental differences in representation rather than implementation details.

---

## 8. Key Findings

From the full set of experiments, we conclude:

- Edge constraints improve geometry but introduce shading artifacts
- Diffusion improves perceptual quality but does not reconstruct true texture
- Purely texture-less planar regions remain a bottleneck for 3DGS
- Many issues cannot be solved by parameter tuning alone

---

## 9. Limitations and Future Work

Potential future directions include:
- explicit planar modeling,
- cross-view texture consistency constraints,
- hybrid GS–NeRF representations.

---

## 10. Hardware and Practical Constraints

All experiments were conducted on a single consumer-grade machine:

- GPU: NVIDIA RTX 4060 (Laptop, 8GB VRAM)
- CPU: Intel i7-class mobile CPU
- RAM: 32GB
- OS: Ubuntu Linux

Due to limited GPU memory, several experiments were run with:
- reduced resolution,
- CPU-based data loading,
- restricted densification settings.

Example constrained training command:

```bash
python train_changed.py \
  --iterations 30000 \
  --densify_until_iter 7000 \
  --densify_grad_threshold 0.0008 \
  --densification_interval 200 \
  --data_device cpu
```
---

OOM constraints influenced both model design and experimental choices.
11. Repository Structure

gaussian-splatting/
├── train.py                # original 3DGS training
├── train_changed.py        # modified training experiments
├── train_with_detail.py    # residual / diffusion experiments
├── traindiff/              # diffusion / residual modules
├── docs/                   # detailed reports
└── ...

12. Notes

This repository intentionally documents both successes and failures.
Understanding where current methods break is considered a core outcome of this project.
