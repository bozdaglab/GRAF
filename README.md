# GRAF
Graph Attention-aware Fusion Networks

A large number of real-world networks include multiple types of nodes and edges. Graph Neural Network (GNN) emerged as a deep learning framework to generate node and graph embeddings for downstream machine learning tasks. However, popular GNN-based architectures operate on single homogeneous networks. Enabling them to work on multiple networks brings additional challenges due to the heterogeneity of the networks and the multiplicity of the existing associations. In this study, we present a computational approach named GRAF (Graph Attention-aware Fusion Networks) utilizing GNN-based approaches on multiple networks with the help of attention mechanisms and network fusion. Using attention-based neighborhood aggregation, GRAF learns the importance of each neighbor per node (called *node-level attention*) followed by the importance of association (called *association-level attention*). Then, GRAF processes a network fusion step weighing each edge according to learned node- and association-level attentions. Considering that the fused network could be a highly dense network with many weak edges depending on the given input networks, we included an edge elimination step with respect to edges' weights. Finally, GRAF utilizes Graph Convolutional Network (GCN) on the fused network and incorporates node features on graph-structured data for a node classification or a similar downstream task. To demonstrate GRAF’s generalizability, we applied it to four datasets from different domains and observed that GRAF outperformed or was on par with the baselines, state-of-the-art methods, and its own variations for each node classification task. 

To learn more about GRAF, read our paper at: [https://doi.org/10.48550/arXiv.2303.16781](https://doi.org/10.48550/arXiv.2303.16781)

---


## How to run GRAF?

Use `GRAF.py` to run GRAF.

Example run:
- `python GRAF.py`: runs GRAF with 'sampledata'

Sample console output:
``` > python GRAF.py
GRAF is running..
Association-based attentions in order: [0.24, 0.22, 0.28, 0.26]
GRAF with 100% of edges: Macro F1: 0.304+-0.031, Accuracy: 0.406+-0.038, Weighted F1 0.452+-0.044. Time: 10.3 seconds/10 runs.
GRAF with 90% of edges: Macro F1: 0.313+-0.022, Accuracy: 0.41+-0.028, Weighted F1 0.448+-0.033. Time: 9.9 seconds/10 runs.
GRAF with 80% of edges: Macro F1: 0.32+-0.027, Accuracy: 0.406+-0.026, Weighted F1 0.428+-0.024. Time: 9.9 seconds/10 runs.
GRAF with 70% of edges: Macro F1: 0.306+-0.023, Accuracy: 0.398+-0.022, Weighted F1 0.435+-0.034. Time: 7.5 seconds/10 runs.
GRAF with 60% of edges: Macro F1: 0.32+-0.018, Accuracy: 0.406+-0.024, Weighted F1 0.438+-0.029. Time: 6.3 seconds/10 runs.
GRAF with 50% of edges: Macro F1: 0.304+-0.028, Accuracy: 0.387+-0.021, Weighted F1 0.412+-0.029. Time: 6.2 seconds/10 runs.
GRAF with 40% of edges: Macro F1: 0.308+-0.015, Accuracy: 0.402+-0.017, Weighted F1 0.43+-0.02. Time: 4.7 seconds/10 runs.
GRAF with 30% of edges: Macro F1: 0.29+-0.033, Accuracy: 0.38+-0.029, Weighted F1 0.408+-0.027. Time: 3.5 seconds/10 runs.
GRAF with 20% of edges: Macro F1: 0.304+-0.018, Accuracy: 0.398+-0.021, Weighted F1 0.435+-0.029. Time: 2.6 seconds/10 runs.
GRAF with 10% of edges: Macro F1: 0.296+-0.019, Accuracy: 0.364+-0.021, Weighted F1 0.391+-0.024. Time: 1.6 seconds/10 runs.
GRAF is done.
```

### Data format: 
- Sample data is under the folder *sampledata*.
- Sample attentions obtained from HAN are under the folder *GRAF_results*. 
- Output files are also under *GRAF_results* folder. The name of folders stands for 'dataName_learningRate_hiddenSize_earlyStoppingPatience'. 
- Under that folder, there are multiple sub folders, each was named as GRAF_prob_x, where x is the % of the edges kept. We put sample results for x = 1 (100%) and x = 0.9 (90%). Each subfolders have '_results.xlsx' keeping evaluation metrics for individual 10 runs and corresponding embeddings (in case they are needed for another downstream task).

---


Relevant package versions in the environment:
```
# Name                    Version                   Build  Channel
cpuonly                   2.0                           0    pytorch
numpy                     1.19.2           py36hadc3359_0
pandas                    1.1.5                    pypi_0    pypi
pickle5                   0.0.12                   pypi_0    pypi
pip                       21.3.1                   pypi_0    pypi
python                    3.6.13               h3758d61_0
python-dateutil           2.8.2                    pypi_0    pypi
pytorch                   1.10.2              py3.6_cpu_0    pytorch
pytorch-mutex             1.0                         cpu    pytorch
scikit-learn              0.24.2                   pypi_0    pypi
torch-geometric           2.0.3                    pypi_0    pypi
torch-scatter             2.0.9                    pypi_0    pypi
torch-sparse              0.6.12                   pypi_0    pypi
torchaudio                0.10.2                 py36_cpu  [cpuonly]  pytorch
torchvision               0.11.3                 py36_cpu  [cpuonly]  pytorch
```
