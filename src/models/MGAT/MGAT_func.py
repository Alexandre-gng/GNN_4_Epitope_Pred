import torch
import numpy as np


def load_branch_weights(branch_model, pt_path):
    """
    Charge les poids d'un GATv2Net pré-entraîné dans une GATv2Branch
    en ignorant la couche de classification.
    """
    # Charger le dictionnaire sauvegardé
    checkpoint = torch.load(pt_path, map_location='cpu')
    
    # Parfois le state_dict est enveloppé dans un objet, on vérifie
    if 'model_state_dict' in checkpoint:
        saved_state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        saved_state_dict = checkpoint['state_dict']
    else:
        saved_state_dict = checkpoint

    # Récupérer le dictionnaire de la branche actuelle
    branch_dict = branch_model.state_dict()

    # Filtrer : On ne garde que les clés qui existent dans la branche 
    # (Cela exclut automatiquement 'classifier.weight' et 'classifier.bias')
    filtered_dict = {k: v for k, v in saved_state_dict.items() if k in branch_dict}

    # Charger les poids
    branch_model.load_state_dict(filtered_dict)
    
    print(f"✅ Successfully loaded {len(filtered_dict)} layers from {pt_path}")
    print(f"❌ Ignored keys: {[k for k in saved_state_dict.keys() if k not in branch_dict]}")