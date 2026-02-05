import torch

from GAT import GATv2Net
from GAT_func import evaluate, evaluate_w_threshold, infer_edge_dim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Load the test graph dataset
epitope3d_test = torch.load("C:\\Users/BISITE/Desktop/GNN_CoPiPred/data/graph_lists/epitope3d/epitope3d_TEST.pt")
print(f"Loaded {len(epitope3d_test)} epitope3d test graphs.")
print(f"Example graph data: {epitope3d_test[0]}")

# Load the model
model_path = "C:\\Users\\BISITE\\Desktop\\GNN_CoPiPred\\src\\models\\trained_models\\G\\model_G.pth"
checkpoint = torch.load(model_path, map_location=device)
print("checkpoint keys:", checkpoint.keys())
config = checkpoint['config']
print(f"Model configuration: {config}")

# Infer edge dimension from the dataset
edge_dim = infer_edge_dim(epitope3d_test)
print(f"Inferred edge_dim: {edge_dim}")

# Use edge_dim from checkpoint config if available, otherwise infer from dataset
checkpoint_edge_dim = getattr(config, 'edge_dim', None)
print(f"Checkpoint edge_dim: {checkpoint_edge_dim}, Inferred edge_dim: {edge_dim}")

model = GATv2Net(
    input_dim=1792,
    hidden_dim=config.hidden_dim,
    output_dim=1,
    num_layers=config.num_layers,
    heads=config.heads,
    concat=True,
    residual=True,
    dropout=config.dropout,
    edge_dim=checkpoint_edge_dim  # Use the edge_dim from saved config
).to(device)
print("state dict keys:", model.state_dict().keys())
model.load_state_dict(checkpoint['model_state_dict'])

criterion = torch.nn.BCEWithLogitsLoss()

abg_loss, auc_pr, auc_roc = evaluate(model, loader=epitope3d_test, criterion=criterion, device=device, use_edge_attr=False)
print(f"Test ABG Loss: {abg_loss:.4f}, AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}")


avg_loss, auc_pr, auc_roc, mcc, acc, f1_score_val = evaluate_w_threshold(model, loader=epitope3d_test, threshold=config.best_threshold, criterion=criterion, device=device, use_edge_attr=False)
print(f"Test Loss: {avg_loss:.4f} | Test AUC-PR: {auc_pr:.4f} | Test AUC-ROC: {auc_roc:.4f} | Test MCC: {mcc:.4f} | Test Accuracy: {acc:.4f} | Test F1 Score: {f1_score_val:.4f}")