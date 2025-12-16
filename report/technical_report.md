Technical Report
An Empirical Study on 3D Gaussian Splatting for Indoor Scene Reconstruction

From Simulation Failures to Real-World Boundary-Aware Improvements

1. Project Motivation and Scope

3D Gaussian Splatting (3DGS) has demonstrated impressive real-time rendering quality and sharp geometric boundaries in controlled benchmarks.
However, its behavior in realistic indoor environments—characterized by large texture-less regions, planar surfaces, and illumination variation—remains insufficiently understood.

This project aims to:

systematically reproduce and stress-test 3DGS under increasingly realistic conditions,

identify failure modes specific to indoor scenes,

and explore targeted engineering modifications, with a particular focus on
background boundaries and texture-less (pure-color) regions.

Rather than proposing a new representation, we focus on empirical analysis, controlled modifications, and failure interpretation.

2. Simulation-Based Experiments and Their Limitations

We initially attempted to validate and study 3DGS entirely in simulation, under the assumption that controllable environments would simplify analysis.
In practice, this assumption proved incorrect.

2.1 Simulation Environment I: Minimal Scene with Pure-Color Geometry
Setup

Simple simulated scene consisting of:

axis-aligned boxes,

planar walls,

pure-color materials.

Static lighting.

Synthetic camera trajectories.

Observed Failure

COLMAP failed to extract a sufficient number of stable feature points:

Pure-color surfaces produced no reliable keypoints

SfM reconstruction either:

failed entirely, or

collapsed into degenerate camera poses.

Conclusion

Although visually simple, this environment is structurally unsuitable for SfM-based pipelines.
This highlights a critical constraint of 3DGS:

3DGS fundamentally depends on the success of SfM, even if the final representation is non-mesh-based.

2.2 Simulation Environment II: Complex Indoor Scene with Robots

To address the lack of features, we constructed a second simulated environment.

Setup

Multiple articulated robot models

Simple indoor layout (walls, floor, furniture)

Increased geometric complexity

More varied camera viewpoints

New Issues Encountered

Despite added geometry, we encountered several new problems:

Camera placement limitations

Restricted baselines due to collision constraints

Many views dominated by walls or floor

Background feature sparsity

Large planar regions remained texture-less

Feature points clustered on robot models only

Insufficient depth variation

Limited parallax across views

Poor depth estimation for background planes

Outcome

SfM reconstructions were unstable, and resulting 3DGS training:

either diverged,

or converged to visually implausible solutions.

Decision

At this point, further simulation-based attempts were deemed unproductive for studying realistic indoor behavior.
We therefore transitioned to real-world data acquisition.

3. Real-World Data Collection and Progressive Dataset Refinement
3.1 First Real-World Dataset: Initial Capture (Failure Case)
Capture Setup

Handheld camera

Indoor room environment

Limited movement space

Natural indoor lighting

Results

Reconstruction technically converged

However:

large background regions appeared extremely blurry,

wall–floor boundaries were poorly defined,

geometry was unstable in texture-less areas.

Analysis

Key contributing factors:

small camera baseline,

limited angular diversity,

illumination variation,

dominance of planar, low-texture regions.

This dataset is preserved as a documented failure case, as it reflects realistic constraints in casual capture scenarios.

3.2 Second Real-World Dataset: Refined Capture Strategy

Based on the above analysis, a second dataset was collected with:

improved camera coverage,

more consistent exposure,

deliberate inclusion of structural edges,

better depth variation where possible.

This dataset serves as the primary evaluation dataset for all subsequent modifications.

4. Data Processing and SfM Considerations

All datasets were processed through a standard SfM pipeline:

COLMAP feature extraction and matching

Camera pose estimation

Coordinate normalization

Validation of reconstruction consistency

Several early failures were traced back not to 3DGS itself, but to:

poor camera pose initialization,

degenerate feature distributions.

This reinforces that data processing quality critically bounds achievable 3DGS results.


5. Core Problem: Background Boundaries and Texture-Less Regions

Through the above experiments, a recurring failure pattern emerged:

walls, floors, and large planar surfaces:

lack discriminative texture,

dominate image area,

exhibit weak depth cues.

This leads to:

ambiguous Gaussian placement,

unstable opacity learning,

blurred boundaries between planar regions.

This observation directly motivated the core modifications of this project.

6. Modifications to the 3DGS Pipeline
6.1 Transparency and Opacity Regularization
Motivation

Background-dominated regions exhibited:

excessive opacity accumulation,

noisy Gaussian stacking.

Modification

Introduced additional regularization terms on opacity

Penalized unstable transparency in low-texture regions

Result

Improved background stability

Reduced floating artifacts

Slight loss of fine detail in some regions

![assets/charts/customdata//comparison-5.png](../assets/charts/customdata//comparison-5.png)

6.2 Densification Threshold Experiments

We systematically varied:

gradient thresholds,

densification schedules.

Observations

Lower thresholds:

sharper boundaries,

but increased noise and shading artifacts.

Higher thresholds:

cleaner appearance,

but loss of geometric definition.

Conclusion

Densification alone cannot resolve texture ambiguity in planar regions.
![assets/charts/customdata//comparison-2.png](../assets/charts/customdata//comparison-2.png)

6.3 Edge-Aware Background and Pure-Color Region Handling (Core Contribution)
Motivation

Standard 3DGS treats all regions uniformly, ignoring the semantic distinction between:

object boundaries,

planar background transitions,

pure-color regions.

Approach

We introduced edge-aware constraints to:

emphasize wall–floor intersections,

reinforce background structural boundaries,

stabilize Gaussian placement in pure-color areas.

Results

Significantly sharper wall corners and planar intersections

Improved geometric clarity

Emergence of shadow-like artifacts due to over-constrained opacity

Interpretation

This experiment demonstrates a key trade-off:

Enforcing geometric boundaries improves structure but can distort photometric realism.

![assets/charts/customdata//comparison-3.png](../assets/charts/customdata//comparison-3.png)


7. Diffusion-Based Refinement

To mitigate perceptual artifacts introduced by hard constraints, we applied diffusion-based refinement.

Observed Effects

Reduced noise and blotchy artifacts

Cleaner, more photo-like appearance

Weaker boundary sharpness

Key Limitation

Diffusion does not recover fine repetitive textures (e.g., floor patterns), indicating that such details are limited by the underlying representation and available geometric information.

![assets/charts/customdata//comparison-4.png](../assets/charts/customdata//comparison-4.png)

8. Comparison with NeRF-Based Methods (Instant-NGP)

Using the same datasets, we evaluated Instant-NGP.

Observations

NeRF better preserves continuous textures

3DGS excels at boundary sharpness and efficiency

This comparison suggests that the remaining artifacts in 3DGS are representation-limited, not merely optimization-related.

9. Hardware Constraints and Training Configuration
Hardware

GPU: NVIDIA RTX 4060 (8GB VRAM)

RAM: 32GB

OS: Ubuntu Linux

Constraints

Due to limited GPU memory:

reduced image resolution,

CPU-based data loading,

constrained densification.

Example training command:

python train.py \
  -s data/custom_scene \
  -m output/scene_gs \
  --iterations 30000 \
  --densify_until_iter 7000 \
  --densify_grad_threshold 0.0008 \
  --densification_interval 200 \
  --data_device cpu

10. Key Findings and Limitations
Findings

Background boundaries require explicit handling

Pure-color planar regions expose 3DGS limitations

Diffusion improves perception, not geometry

Limitations

No explicit planar modeling

No texture-space consistency

Hardware constraints

11. Summary

This project documents a complete empirical journey:

from simulation failures,

through real-world data challenges,

to targeted boundary-aware improvements.

The results highlight both the strengths and structural limits of 3D Gaussian Splatting in indoor environments.