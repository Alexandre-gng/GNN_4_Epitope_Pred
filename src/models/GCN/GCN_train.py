import torch

from GCN import PyGGCNModel
from models.GAT import FOCAL_LOSS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

checkpoint = torch.load("src\\models\\trained_models\\E\\best_model_GCN_E_MCC_0.181.pth", map_location=device)

config = checkpoint['config']
print(f"Model configuration: {config}")

model = PyGGCNModel(
    input_dim=config.,
    hidden_dim=config.n_hidden,
    output_dim=1,
    num_layers=config.num_layers,
    dropout=config.dropout,
    edge_dim=config.edge_dim
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

criterion = FOCAL_LOSS(alpha=config.alpha, gamma=config.gamma, reduction='mean')

