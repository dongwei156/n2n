# N2N
This project involves the code and supplementary materials of paper "Node Representation Learning in Graph via Node-to-Neighbourhood Mutual Information Maximization".

# Overview/Abstract
The key towards learning informative node representations in graphs lies in how to gain contextual information from the neighbourhood. In this work, we present a simple-yet-effective self-supervised node representation learning strategy via directly maximizing the mutual information between the hidden representations of nodes and their neighbourhood, which can be theoretically justified by its link to graph smoothing. Following InfoNCE, our framework is optimized via a surrogate contrastive loss, where the positive selection underpins the quality and efficiency of representation learning. To this end, we propose a topology-aware positive sampling strategy, which samples positives from the neighbourhood by considering the structural dependencies between nodes and thus enables positive selection upfront. In the extreme case when only one positive is sampled, we fully avoid expensive neighbourhood aggregation. Our methods achieve promising performance on various node classification datasets. It is also worth mentioning by applying our loss function to MLP based node encoders, our methods can be orders of faster than existing solutions.

# Citation
If you are use this code for you research, please cite our paper.

    @misc{dong2022node,
          title={Node Representation Learning in Graph via Node-to-Neighbourhood Mutual Information Maximization}, 
          author={Wei Dong and Junsheng Wu and Yi Luo and Zongyuan Ge and Peng Wang},
          year={2022},
          eprint={2203.12265},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }

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

    python unsupervised_representation_learning_main.py --dataset Cora
    
# NF-N2N
This project involves the code and supplementary materials of paper "Self-Supervised Node Representation Learning via Node-to-Neighbourhood Alignment", which is partially based on our conference paper "Node Representation Learning in Graph via Node-to-Neighbourhood Mutual Information Maximization".

# Overview/Abstract
Self-supervised node representation learning aims to learn node representations from unlabelled graphs that rival the supervised counterparts. The key towards learning informative node representations lies in how to effectively gain contextual information from the graph structure. In this work, we present simple-yet-effective self-supervised node representation learning via aligning the hidden representations of nodes and their neighbourhood. Our first idea achieves such node-to-neighbourhood alignment by directly maximizing the mutual information between their representations, which, we prove theoretically, plays the role of graph smoothing. Our framework is optimized via a surrogate contrastive loss and a Topology-Aware Positive Sampling (TAPS) strategy is proposed to sample positives by considering the structural dependencies between nodes, which enables offline positive selection. Considering the excessive memory overheads of contrastive learning, we further propose a negative-free solution, where the main contribution is a Graph Signal Decorrelation (GSD) constraint to avoid representation collapse and over-smoothing. The GSD constraint unifies some of the existing constraints and can be used to derive new implementations to combat representation collapse. By applying our methods on top of simple MLP-based node representation encoders, we learn node representations that achieve promising node classification performance on a set of graph-structured datasets from small- to large-scale.

# Citation
If you are use this code for you research, please cite our paper.

    @article{dong2023self,
      title={Self-Supervised Node Representation Learning via Node-to-Neighbourhood Alignment},
      author={Dong, Wei and Yan, Dawei and Wang, Peng},
      journal={arXiv preprint arXiv:2302.04626},
      year={2023}
    }
    
# Dependencies
* tensorflow == 2.6.0
* numpy == 1.21.2
* scikit-learn == 1.0.1
* heapq == 3.10.2
* scipy == 1.7.1

# Run on Cora
Running NF-N2N (W) is followed as:

    python negative_free_unsupervised_representation_learning_main.py --dataset Cora --contrast-loss distance --batch-normalization Schur_ZCA

Running NF-N2N (WA) is followed as:

    python negative_free_unsupervised_representation_learning_main.py --dataset Cora --contrast-loss distance+auto-correlation --batch-normalization BN
    
Running NF-N2N (WC) is followed as:

    python negative_free_unsupervised_representation_learning_main.py --dataset Cora --contrast-loss cross-correlation --batch-normalization BN
    
# Run on Feather-Lastfm (without node features)
Running NF-N2N (W) is followed as:

    python large_negative_free_unsupervised_representation_learning_main.py --dataset Feather-Lastfm --contrast-loss distance --batch-normalization Schur_ZCA

Running NF-N2N (WA) is followed as:

    python large_negative_free_unsupervised_representation_learning_main.py --dataset Feather-Lastfm --contrast-loss distance+auto-correlation --batch-normalization BN
    
Running NF-N2N (WC) is followed as:

    python large_negative_free_unsupervised_representation_learning_main.py --dataset Feather-Lastfm --contrast-loss cross-correlation --batch-normalization BN
