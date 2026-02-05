import os
from unittest import loader

from IPython import embed
from sklearn.model_selection import train_test_split
import torch

from GAT import GATv2Net
from GAT_func import TrainingConfig, evaluate_w_threshold, infer_edge_dim, run_cross_validation, train_n_epochs, find_best_threshold, evaluate

from torch_geometric.loader import DataLoader

from FOCAL_LOSS import FocalLoss


# Config
PROJECT_ROOT = os.getcwd()
print(f"Project root directory: {PROJECT_ROOT}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Epitope3d dataset
# Spatial 1 (roh1)
roh1_train = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\spatial_1\\epitope3d_TRAIN.pt"))
roh1_test = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\spatial_1\\epitope3d_TEST.pt"))
# Spatial 2 (KNN) (roh2)
roh2_train = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\spatial_2\\epitope3d_TRAIN.pt"))
roh2_test = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\spatial_2\\epitope3d_TEST.pt"))
print(f"Loaded {len(roh1_train)} epitope3d train graphs.")
print(f"Loaded {len(roh1_test)} epitope3d test graphs.")
# Sequential 1 (roh3)
roh3_train = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\sequential_1\\epitope3d_TRAIN.pt"))
roh3_test = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\sequential_1\\epitope3d_TEST.pt"))
print(f"Loaded {len(roh3_train)} epitope3d train graphs.")
print(f"Loaded {len(roh3_test)} epitope3d test graphs.")
# Sequential 2 (roh4)
roh4_train = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\sequential_2\\epitope3d_TRAIN.pt"))
roh4_test = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\sequential_2\\epitope3d_TEST.pt"))
print(f"Loaded {len(roh4_train)} epitope3d train graphs.")
print(f"Loaded {len(roh4_test)} epitope3d test graphs.")




# ====== Fixed hyperparameters ======
config = TrainingConfig(
    epochs=150,
    batch_size=8,
    learning_rate=1e-5,
    weight_decay=1e-8,
    num_folds=1,
    hidden_dim=128,
    heads=8,
    num_layers=8,
    dropout=0.3,
    patience = 20,
    best_threshold = None,
)

# Name - Train dataset - Test dataset - use_edge_attr - Focal Loss - embedding_type
model_configs = [
    # spatial 1
    ("T0", roh1_train, roh1_test, False, True, "ESM2+ESMIF1"),
    ("T1", roh1_train, roh1_test, False, False,"ESM2+ESMIF1"),
    ("T2", roh1_train, roh1_test, True, False,"ESM2+ESMIF1"),
     # spatial 2
    ("T3", roh2_train, roh2_test, False, False, "ESM2+ESMIF1"),
    ("T4", roh2_train, roh2_test, True, False, "ESM2+ESMIF1"),
    # sequential 1
    ("T5", roh3_train, roh3_test, False, False, "ESM2+ESMIF1"),
    ("T6", roh3_train, roh3_test, True, False, "ESM2+ESMIF1"),
    # sequential 2
    ("T7", roh4_train, roh4_test, False, False, "ESM2+ESMIF1"),
    ("T8", roh4_train, roh4_test, True, False, "ESM2+ESMIF1"),
]



# Test n features
input_dim = roh1_train[0].num_node_features
print("Feature dim:", input_dim)

for name, train_dataset, test_dataset, use_edge_attr, use_focal_loss, embedding_type in model_configs:
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
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    else :
        criterion = torch.nn.BCEWithLogitsLoss()

    best_model_state = train_n_epochs(
			model,
			train_loader,
			test_loader,
			optimizer,
			criterion,
            name,
            device,
			use_edge_attr,
			config.epochs,
            config.patience,
		)
    
    best_threshold = find_best_threshold(model, val_loader=test_dataset, device=device, use_edge_attr=use_edge_attr)
    config.best_threshold = best_threshold
    print(f"="*20)
    print(config)

    loss, auc_pr, auc_roc, mcc, acc, f1_score_val = evaluate_w_threshold(model, test_dataset, best_threshold, criterion, device, use_edge_attr)
    print(f"Test Loss: {loss:.4f} | Test AUC-PR: {auc_pr:.4f} | Test AUC-ROC: {auc_roc:.4f} | Test MCC: {mcc:.4f} | Test Accuracy: {acc:.4f} | Test F1 Score: {f1_score_val:.4f}")

    model.load_state_dict(best_model_state)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        }

    torch.save(checkpoint, f'model_{name}.pth')
