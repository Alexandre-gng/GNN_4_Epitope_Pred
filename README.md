# Comparative Study of GNNs for Conformational B-Cell Epitope Prediction

Technical report: [link](https://github.com/Alexandre-gng/GNN_4_Epitope_Pred/blob/main/Study_of_4_GNNs_models_for_Conformational_Epitope_Pred.pdf)

This repository presents a comparative study of four Graph Neural Network (GNN) architectures for predicting conformational epitopes in proteins. The goal is to evaluate, under a consistent experimental setup, which approaches generalize best on Epitope3D.

This work was done during an internship at University of Salamanca
## Project Objective

Conformational epitopes are the majority of B-cell epitopes and remain difficult to detect automatically. This work compares several GNN families, along with graph and loss-function variants, to identify the factors that most impact performance.

## Compared Models (Key Specificity of Each Model)

### GCN
GCN uses a simple and stable neighborhood aggregation, making it a clear and fast baseline to train. Its strength is simplicity, but it captures neighbor importance less precisely.

### GATv2
GATv2 dynamically learns which neighbors are the most useful through an attention mechanism. In practice, this selectivity gives it the best overall results in this study.

### MGAT
MGAT combines multiple graph views (spatial and sequential) to fuse complementary information. The idea is powerful, but in this benchmark the multi-view fusion did not outperform a strong single-view GATv2.

### EGNN
EGNN explicitly models 3D geometry and preserves spatial consistency during message passing. This property makes it highly relevant for structural tasks, with solid performance depending on the graph type.

## Data

The project relies on the Epitope3D dataset and its train/validation/test splits. Each protein residue is represented with combined ESM-2 and ESM-IF1 embeddings.

## Evaluated Variants

- Graph construction: distance-threshold (spatial) vs K-nearest neighbors (KNN)
- With or without distance as edge information
- BCE loss vs Focal Loss
- EGNN variant with attention

## Main Takeaways

- Attention-based models, especially GATv2, are the most competitive.
- KNN graphs often help non-attention models.
- Adding raw distance on edges does not provide robust improvement.
- Focal Loss did not show consistent gains without deeper tuning.

## Repository Structure

- `src/data/`: data loading, embeddings, and graph construction
- `src/models/GCN/`: GCN training and evaluation
- `src/models/GAT/`: GATv2 training and evaluation
- `src/models/EGNN/`: EGNN training and evaluation (with variants)
- `src/models/MGAT/`: multi-view training and evaluation
- `config/`: experiment configuration
- `data/epitope3d/`: data, structures, and pre-generated graphs
- `Latex/`: detailed scientific report

## Installation

Prerequisites:
- Python 3.8+
- GPU environment recommended (CUDA) to accelerate training

Steps:

```bash
git clone https://github.com/Alexandre-gng/GNN_4_Epitope_Pred.git
cd GNN_4_Epitope_Pred
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Training and validation scripts are organized by architecture in `src/models/`. Data paths and main hyperparameters are configured in `config/config.yaml`.

## Current Limitations

- Antigen-only approach: no explicit antibody-antigen specificity
- Limited dataset size for definitive conclusions
- Some variant hyperparameters (especially for Focal Loss) are still underexplored

## Contact

Alexandre Guenegan  
Efrei Paris, Universite Pantheon-Assas
