from dataclasses import dataclass, asdict
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from MGAT import MultiView_GAT
from MGAT_func import TrainingConfig
from multiview_dataset import MultiViewDataset

# Get project root: go up 3 levels from this file (MGAT_train.py -> MGAT -> models -> src -> project_root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
print(f"Project root directory: {PROJECT_ROOT}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



# Epitope3d dataset
# Spatial 1 (roh1)
roh1_train = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\spatial_1\\epitope3d_TRAIN.pt"))
roh1_val = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\spatial_1\\epitope3d_TEST.pt"))
roh1_blind_test = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\spatial_1\\epitope3d_VAL.pt"))
print(f"Loaded train : {len(roh1_train)}, val : {len(roh1_val)}, test : {len(roh1_blind_test)} epitope3d graphs.")
# Spatial 2 (KNN) (roh2)
roh2_train = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\spatial_2\\epitope3d_TRAIN.pt"))
roh2_val = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\spatial_2\\epitope3d_TEST.pt"))
roh2_blind_test = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\spatial_2\\epitope3d_VAL.pt"))
print(f"Loaded train : {len(roh2_train)}, val : {len(roh2_val)}, test : {len(roh2_blind_test)} epitope3d graphs.")
# Sequential 1 (roh3)
roh3_train = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\sequential_1\\epitope3d_TRAIN.pt"))
roh3_val = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\sequential_1\\epitope3d_TEST.pt"))
roh3_blind_test = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\sequential_1\\epitope3d_VAL.pt"))
print(f"Loaded train : {len(roh3_train)}, val : {len(roh3_val)}, test : {len(roh3_blind_test)} epitope3d graphs.")
# Sequential 2 (roh4)
roh4_train = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\sequential_2\\epitope3d_TRAIN.pt"))
roh4_val = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\sequential_2\\epitope3d_TEST.pt"))
roh4_blind_test = torch.load(os.path.join(PROJECT_ROOT, "data\\epitope3d\\graph_list\\sequential_2\\epitope3d_VAL.pt"))
print(f"Loaded train : {len(roh4_train)}, val : {len(roh4_val)}, test : {len(roh4_blind_test)} epitope3d graphs.")
print("roh3_train keys:", roh3_train[0].keys())


config = TrainingConfig(
    epochs=50,
    batch_size=8,
    learning_rate=1e-5,
    weight_decay=1e-8,
    num_folds=10,
    hidden_dim=128,
    heads=4,
    num_layers=8,
    dropout=0.0,
    edge_dim=None,
    reg_lambda = 0.005,
    patience = 10,
    alpha = 0.25,
    gamma = 2.0,
    best_threshold = None,
)

# Create multi-view datasets by combining all 4 views
train_dataset = MultiViewDataset([roh1_train, roh2_train, roh3_train, roh4_train])
val_dataset = MultiViewDataset([roh1_val, roh2_val, roh3_val, roh4_val])
blind_test_dataset = MultiViewDataset([roh1_blind_test, roh2_blind_test, roh3_blind_test, roh4_blind_test])

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
blind_test_loader = DataLoader(blind_test_dataset, batch_size=config.batch_size, shuffle=False)
print(f"len(train_loader.dataset), len(val_loader.dataset), len(blind_test_loader.dataset): {len(train_loader.dataset)}, {len(val_loader.dataset)}, {len(blind_test_loader.dataset)}")
 

# ====== Models versions to test ======
model_configs = [
    # name - train dataset - val dataset - test dataset - edge_dim - focal_loss - embedding_type
    ("MGAT1", train_loader, val_loader, blind_test_loader, 1, True,"ESM2+ESMIF1"),
    ("MGAT2", train_loader, val_loader, blind_test_loader, 1, False,"ESM2+ESMIF1"),
    ("MGAT3", train_loader, val_loader, blind_test_loader, None, True,"ESM2+ESMIF1"),
    ("MGAT4", train_loader, val_loader, blind_test_loader, None, False,"ESM2+ESMIF1"),
]


# Si vous avez vos modèles .pt pré-entraînés, chargez-les ici :
# model.gat_convs[0].load_state_dict(torch.load('view1.pt'))
# model.gat_convs[1].load_state_dict(torch.load('view2.pt'))
# ...

from MGAT_func import train_n_epochs, evaluate, find_best_threshold, evaluate_w_threshold, get_attn_by_view
from MGAT_CV import cross_validate_mgat
from FOCAL_LOSS import FocalLoss
import pandas as pd



# ====== Choose training mode ======
USE_CROSS_VALIDATION = True  # Set to False for simple train/val/test split

if USE_CROSS_VALIDATION:
    # ====== Cross-Validation Training ======
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION MODE")
    print(f"{'='*80}\n")
    
    for name, _, _, _, edge_dim, focal_loss, embedding_type in model_configs:
        print(f"\n{'='*80}")
        print(f"Cross-Validating {name} with embedding type: {embedding_type} | edge_dim: {edge_dim} | focal_loss: {focal_loss}")
        print(f"{'='*80}\n")

        config.edge_dim = edge_dim if edge_dim is not None else 1  # Set edge_dim in config for use in CV training
        
        # Prepare datasets by view: each element is [train, val, test] for one view
        datasets_by_view = [
            [roh1_train, roh1_val, roh1_blind_test],  # View 0: Spatial 1
            [roh2_train, roh2_val, roh2_blind_test],  # View 1: Spatial 2 (KNN)
            [roh3_train, roh3_val, roh3_blind_test],  # View 2: Sequential 1
            [roh4_train, roh4_val, roh4_blind_test],  # View 3: Sequential 2
        ]
        
        config.num_folds = 10


        # Run cross-validation
        oof_metrics = cross_validate_mgat(
            datasets_by_view=datasets_by_view,
            model_name=name,
            config=config,
            device=device,
            project_root=PROJECT_ROOT,
            edge_dim=edge_dim,
            use_focal_loss=focal_loss,
            use_mask=True,
            in_channels=1792,
            out_channels=1,
            num_views=4
        )
        
        print(f"\n{'='*80}")
        print(f"Cross-Validation completed for {name}")
        print(f"OOF AUC-PR: {oof_metrics['oof_metrics']['auc_pr']:.4f}")
        print(f"OOF AUC-ROC: {oof_metrics['oof_metrics']['auc_roc']:.4f}")
        print(f"OOF MCC: {oof_metrics['oof_metrics']['mcc']:.4f}")
        print(f"OOF F1: {oof_metrics['oof_metrics']['f1']:.4f}")
        print(f"OOF Accuracy: {oof_metrics['oof_metrics']['accuracy']:.4f}")
        print(f"{'='*80}\n")

else:
    # ====== Simple Train/Val/Test Split Training ======
    print(f"\n{'='*80}")
    print("SIMPLE TRAIN/VAL/TEST MODE")
    print(f"{'='*80}\n")
    
    # Training code for a list of MGAT models
    for name, train_loader, val_loader, blind_test_loader, edge_dim, focal_loss, embedding_type in model_configs:
        print(f"{'='*80}")
        print(f"Training {name} with embedding type: {embedding_type} | edge_dim: {edge_dim} | focal_loss: {focal_loss}")
        
        # Ensure edge_dim is properly initialized
        edge_dim = edge_dim if edge_dim is not None else 1
        
        model = MultiView_GAT(in_channels=1792, out_channels=1, num_views=4, hidden_channels=config.hidden_dim, num_layers=config.num_layers, heads=config.heads, dropout=config.dropout, edge_dim=edge_dim, reg_lambda=config.reg_lambda).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        if focal_loss:
            criterion = FocalLoss(alpha=config.alpha, gamma=config.gamma, reduction='mean')
        else:
            criterion = nn.BCEWithLogitsLoss()

        best_model_state = train_n_epochs(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            blind_test_loader=blind_test_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
            use_mask=True,
        )

        model.load_state_dict(best_model_state)
        avg_loss, avg_ap, avg_roc = evaluate(model, blind_test_loader, criterion, device, use_mask=True)

        print(f"Test Loss: {avg_loss:.4f} | Test AP: {avg_ap:.4f} | Test ROC-AUC: {avg_roc:.4f}")
        best_threshold = find_best_threshold(model, val_loader, criterion, device, use_mask=True)
        print(f"Best threshold on validation set: {best_threshold:.4f}")
        avg_loss, auc_pr, auc_roc, mcc, f1, acc = evaluate_w_threshold(model, blind_test_loader, criterion, device, best_threshold, use_mask=True)
        print(f"blind Test Loss: {avg_loss:.4f} | blind Test AP: {auc_pr:.4f} | blind Test ROC-AUC: {auc_roc:.4f} | blind Test MCC: {mcc:.4f} | blind Test F1: {f1:.4f} | blind Test Accuracy: {acc:.4f}")

        # Save model    
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': asdict(config),
            'metrics': {
                'loss': avg_loss,
                'ap': avg_ap,
                'roc': avg_roc,
                'mcc': mcc,
                'f1': f1,
                'accuracy': acc
            },
            'threshold': best_threshold,
        }

        metrics_path = PROJECT_ROOT + f"/src/models/trained_model/{name}/{name}_metrics.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        model_path = PROJECT_ROOT + f"/src/models/trained_model/{name}/{name}.pt"
        torch.save(checkpoint, model_path)

        res = get_attn_by_view(model, blind_test_loader, device, use_mask=True)
        print(f"Attention weights by view for {name}:")
        for classe, view in res.items():
            print(f"  Class {classe}:")
            for v in range(len(view)):
                attn = view[v]
                print(f"    View {v}: {attn:.4f}")

        df = pd.DataFrame(res)
        df.to_csv(PROJECT_ROOT + f"/src/models/trained_model/{name}/{name}_attn_by_view.csv", index=False)



"""
import optuna 
from MGAT_tune import objective

for name, train_loader, val_loader, blind_test_loader, use_edge_attr, focal_loss, embedding_type in model_configs:
    print(f"{'='*80}")
    print(f"Training {name} with embedding type: {embedding_type} | use_edge_attr: {use_edge_attr} | focal_loss: {focal_loss}")

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )

    study.optimize(lambda trial: objective(trial, config, train_loader, val_loader, blind_test_loader, device, use_edge_attr=True, use_focal_loss=focal_loss, use_mask=True), n_trials=60, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best params and metrics to a file
    best_params = trial.params
    with open(os.path.join(PROJECT_ROOT, "best_mgat_params.txt"), "w") as f:
        f.write(f"Best trial value (validation AUC-PR): {trial.value}\n")
        f.write("Best hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
"""


"""
from MGAT_func import train_n_epochs, evaluate
from FOCAL_LOSS import FocalLoss

# Training code for a list of MGAT models
for name, train_loader, val_loader, test_loader, use_edge_attr, focal_loss, embedding_type in model_configs:
    print(f"{'='*80}")
    print(f"Training {name} with embedding type: {embedding_type} | use_edge_attr: {use_edge_attr} | focal_loss: {focal_loss}")
    
    model = MultiView_GAT(in_channels=1792, out_channels=1, num_views=4, hidden_channels=config.hidden_dim, num_layers=config.num_layers, heads=config.heads, dropout=config.dropout, reg_lambda=config.reg_lambda).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_model_state = train_n_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device,
        use_edge_attr=use_edge_attr,
        use_mask=True,
    )

    model.load_state_dict(best_model_state)
    avg_loss, avg_ap, avg_roc = evaluate(model, test_loader, criterion, device, use_edge_attr, use_mask=True)
    print(f"Test Loss: {avg_loss:.4f} | Test AP: {avg_ap:.4f} | Test ROC-AUC: {avg_roc:.4f}")
"""