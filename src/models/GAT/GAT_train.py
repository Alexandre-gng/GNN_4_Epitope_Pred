import os
from dataclasses import asdict
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
    epochs=5,
    batch_size=8,
    learning_rate=1e-5,
    weight_decay=1e-8,
    num_folds=2,
    hidden_dim=128,
    heads=8,
    num_layers=8,
    dropout=0.0,
    patience = 20,
    best_threshold = None,
)


# Name - Train dataset - Test dataset - use_edge_attr - Focal Loss - embedding_type
model_configs = [
    # spatial 1
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
    print(f"\n{'='*80}")
    print(f"Training Model: {name}")
    print(f"{'='*80}")
    
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

    optimizer = torch.optim.AdamW(build_model().parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # Prepare criterion
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    best_model_state = train_n_epochs(
        model=build_model(),
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        name_model=name,
        device=device,
        use_edge_attr=use_edge_attr,
        n_epochs=config.epochs,
        patience=config.patience,
        use_mlflow=False,
    )
    

    checkpoint = {
        'model_state_dict': best_model_state.state_dict(),
        'config': asdict(config),
        'metrics': fold_metrics_dict,
        'threshold': fold_threshold,
    }
    fold_model_path = model_dir / f"fold_{fold_idx}_model.pt"
    torch.save(checkpoint, fold_model_path)


    """
    # Run cross-validation (threshold will be computed for each fold)
    print(f"\nRunning k-fold cross-validation...")
    print(f"Note: Best threshold will be computed for each fold by maximizing MCC")
    run_cross_validation(
        train_dataset,
        test_dataset,
        build_model,
        criterion,
        config,
        device,
        use_edge_attr,
        model_name=name,
        seed=42,
    )
    """