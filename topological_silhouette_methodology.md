# Topological Silhouette: Manifold-Aware Margin for Selective Classification

## 1. Motivation

Selective Classification (SC), or classification with a reject option, aims to abstain from making predictions when the model's confidence is low. This is critical for deploying neural networks in high-stakes environments like medical diagnosis or autonomous driving. 

Recent advancements have shown that training networks with **Mixup** significantly improves their ability to estimate uncertainty. Mixup acts as a strong regularizer that prevents overconfidence on out-of-distribution or ambiguous samples. However, effectively extracting this calibrated uncertainty from the model post-training remains an open challenge. 

Existing post-hoc methods in the literature typically fall into two categories:
1. **Logit-Space Methods:** Methods like Maximum Softmax Probability (MSP) or **MaxLogit pNorm** operate on the final linear projection of the network. While `pNorm` has established itself as a robust state-of-the-art metric for ranking (AURC) on Mixup-trained models, it fundamentally relies on the linear classifier. Because it relies purely on magnitudes without probability normalization, it often severely degrades the model's calibration (Expected Calibration Error).
2. **Feature-Space Methods:** Methods like **ViM (Virtual Logit Matching)** or **Mahalanobis Distance** operate on the penultimate representation layer. While these methods excel at Out-of-Distribution (OOD) detection, they frequently underperform in Selective Classification tasks because they focus on global distribution fitting rather than the relative, localized margin between specific classes.

The motivation for our work is to bridge this gap: **Can we design a purely feature-space method that captures the geometric "sharpness" and margin-awareness of logit-based metrics, thereby exceeding their ranking performance while operating on the uncompressed representation manifold?**

## 2. Research Narrative

The development of our method began by analyzing the mechanism that makes logit-space methods (like `MaxLogit pNorm`) successful, and identifying the information they inherently discard.

### The Role of Neural Collapse and the ETF Filter
Theoretical analysis of representation learning, specifically the phenomenon of **Neural Collapse**, reveals that during terminal phases of training (especially under strong regularization like Mixup), the final classifier matrix converges to a simplex Equiangular Tight Frame (ETF). 

The logit space is simply the result of projecting the features onto these classifier weights. Because the classifier forms an optimal ETF, this projection acts as a mathematical filter—discarding orthogonal variations and focusing purely on linearly separable components. `MaxLogit pNorm` capitalizes on this clean, filtered space by measuring the statistical prominence ($L_p$ norm) of the predicted logit against the rest. 

### The Missing Link: Uncompressed Local Topology
While the ETF filter cleans the signal, it also forces a massive compression of information. It enforces a strict assumption that the data manifolds are perfectly linearly separable and perfectly collapsed to points. 

In reality, at inference time, validation and test features rarely exhibit strict Neural Collapse. They form complex, non-linear, uncollapsed manifolds. By strictly using the logit space, existing methods throw away the rich, localized geometric structure of these representations. On the other hand, existing feature-space methods like `ViM` apply global Principal Component Analysis (PCA) or fit global Gaussian distributions, which smooth over these local, non-linear boundaries.

To surpass the state-of-the-art, we needed a metric that probes the raw geometry of these uncollapsed manifolds directly. In classification, certainty is dictated by the **relative margin**: a highly confident sample must not only be densely surrounded by its predicted class, but it must also be topologically isolated from competing classes.

## 3. Methodology: Topological Silhouette

To directly measure this feature-space margin, we introduce the **Topological Silhouette** method. Inspired by the Silhouette Coefficient used in unsupervised clustering, this method calculates the confidence of a prediction based on the tightness of the sample to its predicted class manifold relative to its proximity to the nearest competing class manifold.

### The Algorithm

Given a test sample's feature vector $h_{test}$, a validation set $H_{val}$ with labels $Y_{val}$, and the base model's predicted probabilities $P_{base}$:

**Step 1: Spherical Projection**
To remove magnitude-based noise and focus purely on angular semantics (which are heavily encouraged by Mixup), we first center the features using the global training mean $\bar{h}$ and project them onto the $L_2$ unit hypersphere:
$$ \tilde{h} = \frac{h - \bar{h}}{||h - \bar{h}||_2} $$

**Step 2: Split-Validation for Calibration**
To tune hyperparameters without overfitting, we split the validation set $H_{val}$ into two halves:
1.  **Reference Set ($H_{ref}$)**: Used as the kNN database.
2.  **Calibration Set ($H_{cal}$)**: Used to evaluate hyperparameter choices.

**Step 3: Class-Wise Distance Estimation**
For a query feature $\tilde{h}_{query}$ (from $H_{cal}$ or the test set), we compute the average Euclidean distance to its $k$-nearest neighbors in the Reference Set belonging to class $c$, for all classes $c \in \{1, \dots, C\}$:
$$ D_c(\tilde{h}_{query}) = \frac{1}{k} \sum_{i=1}^k ||\tilde{h}_{query} - \tilde{h}_{ref, i}^{(c)}||_2 $$

**Step 4: Topological Margin Calculation**
Let $c_{pred}$ be the class predicted by the base model. We define:
*   **$D_{same}$**: Distance to the predicted class manifold: $D_{same} = D_{c_{pred}}$
*   **$D_{diff}$**: Distance to the closest competing class manifold: $D_{diff} = \min_{c \neq c_{pred}} D_c$

The raw Silhouette score is calculated as:
$$ Conf_{topo} = \frac{D_{diff} - D_{same}}{D_{diff} + D_{same} + \epsilon} $$
where $\epsilon$ is a small constant for numerical stability.

**Step 5: Grid Search and Blending**
To combine the complementary strengths of logit-space and feature-space signals, we define the blended confidence as:
$$ Conf_{blend} = \alpha \cdot Conf_{topo} + (1 - \alpha) \cdot Conf_{base} $$
where $Conf_{base}$ is the Maximum Softmax Probability (MSP) or another base confidence score.

We perform a grid search over candidate values for $k$ and $\alpha$:
*   $k \in \{5, 10, 20, 30, 50, \dots\}$
*   $\alpha \in [0.0, 1.0]$

We evaluate each combination on the Calibration Set $H_{cal}$ and select the $(k^*, \alpha^*)$ pair that maximizes the Area Under the Risk-Coverage curve (AURC) or AUROC.

**Step 6: Final Test Evaluation (Refit)**
After finding the optimal $k^*$ and $\alpha^*$:
1.  We merge the split validation set back into a single full reference set $H_{val}$.
2.  For each test sample, we compute $Conf_{topo}$ using the full $H_{val}$ as the kNN reference with $k^*$.
3.  The final confidence score for selective classification is:
    $$ Conf_{final} = \alpha^* \cdot Conf_{topo} + (1 - \alpha^*) \cdot Conf_{base} $$

---

By dynamically tracking the local $k$NN density and blending it with the base classifier's confidence, the Topological Silhouette naturally contours to the true, non-linear shape of the validation embeddings while maintaining the clean filtering of the logit space. It successfully extracts a richer, margin-aware geometric signal than what is available in the linearly compressed logit space alone.
