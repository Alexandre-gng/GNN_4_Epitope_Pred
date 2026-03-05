import torch
import os

from EGNN import EGNN
from EGNN_func import TrainingConfig, evaluate, train_one_epoch, train_n_epochs
from EGNN_CV import cross_validate_egnn


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============ Spatial_1 (roh_1) dataset loading ============
roh1_train = torch.load(os.path.join(PROJECT_ROOT, "data", "epitope3d", "graph_list", "spatial_1", "epitope3d_TRAIN.pt"))
roh1_val = torch.load(os.path.join(PROJECT_ROOT, "data", "epitope3d", "graph_list", "spatial_1", "epitope3d_TEST.pt"))
roh1_blind_test = torch.load(os.path.join(PROJECT_ROOT, "data", "epitope3d", "graph_list", "spatial_1", "epitope3d_VAL.pt"))
print(f"Loaded train : {len(roh1_train)}, val : {len(roh1_val)}, test : {len(roh1_blind_test)} epitope3d graphs.")
# ============ Spatial_2 (roh_2) dataset loading ============
roh2_train = torch.load(os.path.join(PROJECT_ROOT, "data", "epitope3d", "graph_list", "spatial_2", "epitope3d_TRAIN.pt"))
roh2_val = torch.load(os.path.join(PROJECT_ROOT, "data", "epitope3d", "graph_list", "spatial_2", "epitope3d_TEST.pt"))
roh2_blind_test = torch.load(os.path.join(PROJECT_ROOT, "data", "epitope3d", "graph_list", "spatial_2", "epitope3d_VAL.pt"))
print(f"Loaded train : {len(roh2_train)}, val : {len(roh2_val)}, test : {len(roh2_blind_test)} epitope3d graphs.")


graph0 = roh1_train[0]
node_attr = graph0["node_attrs"]
node_coords = graph0["coords"]
edge_index = graph0["edge_index"]
edge_attr = graph0["edge_attr"]

print("train attributes:")
for key in graph0.keys():
    print(f"{key}")

print(f"Node feature dimension: {node_attr.shape[1]}")
print(f"Node coordinate dimension: {node_coords.shape[1]}")
print(f"Edge index shape: {edge_index.shape}, Edge attribute shape: {edge_attr.shape if edge_attr is not None else 'N/A'}")


# ============ Training configuration ============
config = TrainingConfig(
    epochs=50,
    batch_size=8,
    learning_rate=1e-4,
    weight_decay=1e-5,
    num_folds=10,
    num_layers=3,
    hidden_dim=128,
    out_dim=64,
    edge_dim=edge_attr.shape[1],
    dropout=0.1,
    patience=10,
    update_coords=True,
    max_coord_step=1.0,
    max_abs_coord_value=1e4,
)
print(f"Training configuration: {config}")
"""

"""
model_configs = [
    # Model name, train data, val data, blind test data, use focal loss, feature description
    ("EGNN_attn_spatial_1_bce",   roh1_train, roh1_val, roh1_blind_test, False, "ESM2+ESMIF1"),
    ("EGNN_attn_spatial_1_focal", roh1_train, roh1_val, roh1_blind_test, True,  "ESM2+ESMIF1"),    
    ("EGNN_attn_spatial_2_bce",   roh2_train, roh2_val, roh2_blind_test, False, "ESM2+ESMIF1"),
    ("EGNN_attn_spatial_2_focal", roh2_train, roh2_val, roh2_blind_test, True,  "ESM2+ESMIF1"),
]

for model_name, train_data, val_data, blind_test_data, use_focal_loss, feature_desc in model_configs:
    print("\n" + "=" * 80)
    print(f"Training {model_name} with features: {feature_desc} and focal loss: {use_focal_loss}")
    print("=" * 80)
    # ============ Cross-Validation ============
    oof_metrics = cross_validate_egnn(
        train_data=train_data,
        val_data=val_data,
        blind_test_data=blind_test_data,
        model_name=model_name,
        config=config,
        device=device,
        project_root=PROJECT_ROOT,
        use_focal_loss=use_focal_loss,
        use_mask=True,
        in_channels=node_attr.shape[1],
    )

print("\n" + "=" * 80)
print("CROSS-VALIDATION COMPLETE")
print("=" * 80)
print(f"OOF AUC-PR:  {oof_metrics['oof_metrics']['auc_pr']:.4f}")
print(f"OOF AUC-ROC: {oof_metrics['oof_metrics']['auc_roc']:.4f}")
print(f"OOF MCC:     {oof_metrics['oof_metrics']['mcc']:.4f}")
print(f"OOF F1:      {oof_metrics['oof_metrics']['f1']:.4f}")
print(f"OOF Acc:     {oof_metrics['oof_metrics']['accuracy']:.4f}")