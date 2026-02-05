import torch
from torch_geometric.data import Data

import optuna

from 
from models.GAT import GAT
from models.GAT.GAT_func import train, evaluate, train_one_epoch


def objective(trial, train_loader, val_loader, config, device):
    # Hyperparameters à optimiser
    gamma = trial.suggest_int("hidden_dim", 32, 256)
    alpha = trial.suggest_int("num_heads", 1, 8)

    # Créer le modèle avec les hyperparamètres suggérés
    model = GAT(input_dim=config.input_dim, hidden_dim=config., num_heads=alpha, dropout=config.dropout, use_edge_attr=config.use_edge_attr).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Définir l'optimiseur et la fonction de perte
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Entraîner le modèle (utiliser une partie des données pour la validation)
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(model, loader=train_loader, optimizer=optimizer, criterion=criterion, device=device, use_edge_attr=config.use_edge_attr)
        val_loss = evaluate(model, loader=val_loader, criterion=criterion, device=device, use_edge_attr=config.use_edge_attr)
        
        # Enregistrer le meilleur modèle basé sur la perte de validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss