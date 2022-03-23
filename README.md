# N2N
This project involves the code and supplementary materials of paper "Node Representation Learning in Graph via Node-to-Neighbourhood Mutual Information Maximization".

# Overview/Abstract
The key towards learning informative node representations in graphs lies in how to gain contextual information from the neighbourhood. In this work, we present a simple-yet-effective self-supervised node representation learning strategy via directly maximizing the mutual information between the hidden representations of nodes and their neighbourhood, which can be theoretically justified by its link to graph smoothing. Following InfoNCE, our framework is optimized via a surrogate contrastive loss, where the positive selection underpins the quality and efficiency of representation learning. To this end, we propose a topology-aware positive sampling strategy, which samples positives from the neighbourhood by considering the structural dependencies between nodes and thus enables positive selection upfront. In the extreme case when only one positive is sampled, we fully avoid expensive neighbourhood aggregation. Our methods achieve promising performance on various node classification datasets. It is also worth mentioning by applying our loss function to MLP based node encoders, our methods can be orders of faster than existing solutions.

# Citation
If you are use this code for you research, please cite our paper.

# Dependencies
* tensorflow == 2.6.0
* numpy == 1.21.2
* scikit-learn == 1.0.1
* heapq == 3.10.2
* scipy == 1.7.1

# Run
Running N2N-TAPS-1 (JL) is followed as:

    python joint_learning_main.py --dataset Cora

Running N2N-TAPS-1 (URL) is followed as:

    python contrastive_learning_main.py --dataset Cora
