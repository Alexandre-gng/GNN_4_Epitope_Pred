import os
from unittest import loader

from IPython import embed
from sklearn.model_selection import train_test_split
import torch

from GAT import GATv2Net
from GAT_func import TrainingConfig, evaluate_w_threshold, infer_edge_dim, run_cross_validation, train_n_epochs, find_best_threshold, evaluate

from torch_geometric.loader import DataLoader


# Config
PROJECT_ROOT = os.getcwd()
print(f"Project root directory: {PROJECT_ROOT}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



# ====== Load TRAIN graph datasets ======
roh1_esm2_esmIF1_train = torch.load(os.path.join(PROJECT_ROOT, "data\graph_lists\sequential_1\ESM2_ESMIF1//roh1_ESM2_ESMIF1_TRAIN.pt"))
roh2_esm2_esmIF1_train = torch.load(os.path.join(PROJECT_ROOT, "data\graph_lists\sequential_2\ESM2_ESMIF1//roh2_ESM2_ESMIF1_TRAIN.pt"))
print(f"Loaded {len(roh1_esm2_esmIF1_train)} ESM2 + ESMIF1 graphs.")
print(f"Loaded {len(roh2_esm2_esmIF1_train)} ESM2 + ESMIF1 graphs.")
# Spatial
roh3_esm2_esmIF1_train = torch.load(os.path.join(PROJECT_ROOT, "data\graph_lists\spatial_1\ESM2_ESMIF1//roh3_ESM2_ESMIF1_TRAIN.pt"))
roh4_esm2_esmIF1_train = torch.load(os.path.join(PROJECT_ROOT, "data\graph_lists\spatial_2\ESM2_ESMIF1//roh4_ESM2_ESMIF1_TRAIN.pt"))
print(f"Loaded {len(roh3_esm2_esmIF1_train)} ESM2 + ESMIF1 graphs.")
print(f"Loaded {len(roh4_esm2_esmIF1_train)} ESM2 + ESMIF1 graphs.")
# ====== Load TEST graph datasets ======
# Sequential
roh1_esm2_esmIF1_test = torch.load(os.path.join(PROJECT_ROOT, "data\graph_lists\sequential_1\ESM2_ESMIF1//roh1_ESM2_ESMIF1_TEST.pt"))
roh2_esm2_esmIF1_test = torch.load(os.path.join(PROJECT_ROOT, "data\graph_lists\sequential_2\ESM2_ESMIF1//roh2_ESM2_ESMIF1_TEST.pt"))
print(f"Loaded {len(roh1_esm2_esmIF1_test)} ESM2 + ESMIF1 test graphs.")
print(f"Loaded {len(roh2_esm2_esmIF1_test)} ESM2 + ESMIF1 test graphs.")
# Spatial
roh3_esm2_esmIF1_test = torch.load(os.path.join(PROJECT_ROOT, "data\graph_lists\spatial_1\ESM2_ESMIF1//roh3_ESM2_ESMIF1_TEST.pt"))
roh4_esm2_esmIF1_test = torch.load(os.path.join(PROJECT_ROOT, "data\graph_lists\spatial_2\ESM2_ESMIF1//roh4_ESM2_ESMIF1_TEST.pt"))
print(f"Loaded {len(roh3_esm2_esmIF1_test)} ESM2 + ESMIF1 test graphs.")
print(f"Loaded {len(roh4_esm2_esmIF1_test)} ESM2 + ESMIF1 test graphs.")
# ====== Load VAL graph datasets ======
# Sequential
roh1_esm2_esmIF1_val = torch.load(os.path.join(PROJECT_ROOT, "data\graph_lists\sequential_1\ESM2_ESMIF1//roh1_ESM2_ESMIF1_VAL.pt"))
roh2_esm2_esmIF1_val = torch.load(os.path.join(PROJECT_ROOT, "data\graph_lists\sequential_2\ESM2_ESMIF1//roh2_ESM2_ESMIF1_VAL.pt"))
print(f"Loaded {len(roh1_esm2_esmIF1_val)} ESM2 + ESMIF1 val graphs.")
print(f"Loaded {len(roh2_esm2_esmIF1_val)} ESM2 + ESMIF1 val graphs.")
# Spatial
roh3_esm2_esmIF1_val = torch.load(os.path.join(PROJECT_ROOT, "data\graph_lists\spatial_1\ESM2_ESMIF1//roh3_ESM2_ESMIF1_VAL.pt"))
roh4_esm2_esmIF1_val = torch.load(os.path.join(PROJECT_ROOT, "data\graph_lists\spatial_2\ESM2_ESMIF1//roh4_ESM2_ESMIF1_VAL.pt"))
print(f"Loaded {len(roh3_esm2_esmIF1_val)} ESM2 + ESMIF1 val graphs.")
print(f"Loaded {len(roh4_esm2_esmIF1_val)} ESM2 + ESMIF1 val graphs.")



# ====== Fixed hyperparameters ======
config = TrainingConfig(
    epochs=100,
    batch_size=8,
    learning_rate=1e-5,
    weight_decay=1e-8,
    num_folds=1,
    hidden_dim=128,
    heads=8,
    num_layers=8,
    dropout=0.4,
    patience = 40,
    best_threshold = None,
)



# Define model configurations: (name, train_dataset, test_dataset, use_edge_attr, embedding_type)
model_configs = [
    # seq1
    ("G", roh1_esm2_esmIF1_train, roh1_esm2_esmIF1_val, roh1_esm2_esmIF1_test, False, "ESM2+ESMIF1"),
    ("H", roh1_esm2_esmIF1_train, roh1_esm2_esmIF1_val, roh1_esm2_esmIF1_test, True, "ESM2+ESMIF1"),
    # seq2
    ("I", roh2_esm2_esmIF1_train, roh2_esm2_esmIF1_val, roh2_esm2_esmIF1_test, False, "ESM2+ESMIF1"),
    ("J", roh2_esm2_esmIF1_train, roh2_esm2_esmIF1_val, roh2_esm2_esmIF1_test, True, "ESM2+ESMIF1"),
    # spatial 1
    ("K", roh3_esm2_esmIF1_train, roh3_esm2_esmIF1_val, roh3_esm2_esmIF1_test, False, "ESM2+ESMIF1"),
    ("L", roh3_esm2_esmIF1_train, roh3_esm2_esmIF1_val, roh3_esm2_esmIF1_test, True, "ESM2+ESMIF1"),
    # spatial 2
    ("M", roh4_esm2_esmIF1_train, roh4_esm2_esmIF1_val, roh4_esm2_esmIF1_test, False, "ESM2+ESMIF1"),
    ("N", roh4_esm2_esmIF1_train, roh4_esm2_esmIF1_val, roh4_esm2_esmIF1_test, True, "ESM2+ESMIF1"),
]

# Test n features
input_dim = roh1_esm2_esmIF1_train[0].num_node_features
print("Feature dim:", input_dim)

for name, train_dataset, val_dataset, test_dataset, use_edge_attr, embedding_type in model_configs:
    edge_dim = infer_edge_dim(train_dataset) if use_edge_attr else None
    print(f"edge_dim: {edge_dim} _ use_edge_attr: {use_edge_attr}")
    def build_model():
        return GATv2Net(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_layers=config.num_layers,
            heads=config.heads,
            concat=True,
            residual=True,
            dropout=config.dropout,
            edge_dim=edge_dim,
        )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_model_state = train_n_epochs(
			model,
			train_loader,
			val_loader,
			optimizer,
			criterion,
			device,
			use_edge_attr,
			config.epochs,
            config.patience,
		)
    
    best_threshold = find_best_threshold(model, val_loader=val_loader, device=device, use_edge_attr=use_edge_attr)
    config.best_threshold = best_threshold
    print(f"="*20)
    print(config)

    loss, auc_pr, auc_roc, mcc, acc, f1_score_val = evaluate_w_threshold(model, test_loader, best_threshold, criterion, device, use_edge_attr)
    print(f"Test Loss: {loss:.4f} | Test AUC-PR: {auc_pr:.4f} | Test AUC-ROC: {auc_roc:.4f} | Test MCC: {mcc:.4f} | Test Accuracy: {acc:.4f} | Test F1 Score: {f1_score_val:.4f}")

    model.load_state_dict(best_model_state)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        }

    torch.save(checkpoint, f'model_{name}.pth')
