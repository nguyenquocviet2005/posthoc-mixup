# Softmin Class Probability (SCP) for Selective Classification

## Motivation

Existing feature-space confidence methods, most notably Google's Trust Score, rely on a "ratio of distances" or a "margin" based on the closest point in a differing class. Specifically, Trust Score computes the ratio between the distance to the nearest class (other than the predicted one) and the distance to the predicted class. While effective in standard-trained feature spaces where classes form tight, isolated clusters, this approach becomes brittle in models trained with Mixup.

Mixup forces the model to learn on convex combinations of inputs and labels, resulting in a feature space characterized by smooth, continuous transition regions between class clusters. In these regions, a single outlier or a minor shift in the local neighborhood can drastically change the "nearest other class" distance, making Trust Score's min-based operator noisy and unstable. We need a method that respects the continuous nature of Mixup spaces rather than imposing hard, boundary-based metrics.

## Research Narrative

Our investigation began with the "Topological Silhouette" method, which claimed to exploit feature space geometry but was empirically demonstrated to be a monotone transformation of Trust Score combined with generic temperature scaling and MSP blending. Recognizing the lack of genuine novelty and the specific challenges posed by Mixup geometry, we set out to design a method that does not simply borrow OOD or cluster-based metrics but actively respects the multi-class continuum of Mixup spaces.

We hypothesized that instead of reducing the local neighborhood to a binary comparison (Predicted Class vs. Closest Competitor), we should treat the local geometry as a probability distribution over all classes. This led to the development of Softmin Class Probability (SCP). By converting all class distances into a probability distribution, SCP provides a smooth, global view of the local neighborhood, making it robust to the dense interpolation regions typical of Mixup. Empirical results confirmed this hypothesis, showing that SCP consistently outperforms Trust Score and is highly competitive with state-of-the-art logit calibration methods like MaxLogit pNorm+.

## Methodology

Let $\mathcal{D} = \{(f_i, y_i)\}_{i=1}^N$ be the reference (training) set where $f_i \in \mathbb{R}^d$ are features and $y_i \in \{1, \dots, C\}$ are class labels.

For a query feature $f_q$:

1.  **Class-Conditional Distance**: For each class $c \in \{1, \dots, C\}$, we identify the $k$ nearest neighbors in $\mathcal{D}$ that belong to class $c$. Let this set be $\mathcal{N}_k(f_q, c)$. We compute the average distance to these neighbors:
    $$D_c(f_q) = \frac{1}{k} \sum_{f \in \mathcal{N}_k(f_q, c)} \| f_q - f \|_2$$

2.  **Softmin Transformation**: To convert these distances into a proper probability distribution that favors smaller distances, we apply the softmin function with temperature $T$:
    $$P_c(f_q) = \frac{\exp(-D_c(f_q) / T)}{\sum_{c'=1}^C \exp(-D_{c'}(f_q) / T)}$$

3.  **Confidence Estimation**: The final confidence score for the query point is the maximum probability assigned to any class:
    $$\text{Conf}_{SCP}(f_q) = \max_c P_c(f_q)$$

In our experiments, we found that a small neighborhood ($k=5$ or $k=10$) and a sharp temperature ($T=0.1$) yield the best results, suggesting that while all classes should be considered for normalization, the local neighborhood remains the most informative signal.
