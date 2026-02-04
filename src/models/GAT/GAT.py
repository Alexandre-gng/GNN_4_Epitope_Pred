"""
Graph Attention Network v2 (GATv2) implementation for graph neural networks.

This module implements GATv2, an improved version of the Graph Attention Network that allows 
attention heads to learn both the importance of neighbors and the importance of different 
features. It's particularly useful for node-level classification tasks on graphs.

GATv2 improves upon GAT by using multi-head attention with dynamic feature importance.
"""

import torch
from torch_geometric.nn import GATv2Conv

class GATv2Layer(torch.nn.Module):
    """
    A single Graph Attention Network v2 (GATv2) layer.
    
    This layer applies multi-head attention over graph edges, allowing each attention head
    to independently learn which neighbors are most important. Can optionally include:
    - Residual connections for improved gradient flow
    - Dropout for regularization
    - Edge attributes (e.g., distances) to influence attention
    
    Args:
        in_dim: Input feature dimension for each node
        out_dim: Output feature dimension per attention head
        heads: Number of attention heads (run in parallel)
        concat: If True, concatenate head outputs; if False, average them
        residual: If True, add residual connection from input to output
        dropout: Dropout probability applied to attention weights
        negative_slope: Negative slope of the LeakyReLU activation in attention computation
        edge_dim: Dimension of edge attributes (distances, edge features, etc.). None if no edge attributes
    """
    
    def __init__(self ,in_dim: int, out_dim: int, heads: int, concat: bool, residual: bool, dropout: float = 0.6, negative_slope: float = 0.2, edge_dim: int = None):
        super().__init__()
        
        # GATv2 convolutional layer that computes attention and aggregates neighbor features
        self.conv = GATv2Conv(in_channels=in_dim, out_channels=out_dim, heads=heads, concat=concat, dropout=dropout, negative_slope=negative_slope, edge_dim=edge_dim)

        self.concat = concat
        self.residual = residual
        self.dropout = torch.nn.Dropout(dropout)
        self.act = torch.nn.ELU()

        # Calculate output feature dimension based on concatenation strategy
        out_features = out_dim * heads if concat else out_dim
        
        # Optional linear transformation for residual connection
        self.res_connection = torch.nn.Linear(in_dim, out_features) if residual and in_dim != out_features else None


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GATv2 layer.
        
        Args:
            x: Node feature tensor of shape [num_nodes, in_dim]
            edge_index: Graph connectivity in COO format [2, num_edges]
            edge_attr: Optional edge attributes (distances) [num_edges, edge_dim]
        
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Apply GATv2 convolution with optional edge attributes
        if edge_attr is not None:
            out, (edge_index_out, attn) = self.conv(x, edge_index, edge_attr, return_attention_weights=True)
            
        else:
            out, (edge_index_out, attn) = self.conv(x, edge_index, return_attention_weights=True)
        
        out = self.dropout(out)

        if self.residual:
            # Use identity residual if dimensions match, otherwise use linear transformation
            res = x if self.res_connection is None else self.res_connection(x)
            out += res

        out = self.act(out)
        return out, edge_index_out, attn



class GATv2Net(torch.nn.Module):
    """
    Complete Graph Attention Network v2 architecture for node-level prediction tasks.
    
    This network stacks multiple GATv2 layers to learn hierarchical graph representations,
    suitable for tasks like node classification or epitope prediction on protein graphs.
    
    Architecture flow:
    1. Node Encoder: Project input features to hidden dimension
    2. GATv2 Layers: Apply multi-head attention to learn graph structure and features
    3. Classifier: Linear layer to produce final predictions
    
    Args:
        input_dim: Dimension of input node features (e.g., 1280 for ESM2 embeddings)
        hidden_dim: Dimension of hidden features after encoding
        output_dim: Dimension of output predictions (default=1 for binary classification)
        num_layers: Number of GATv2 layers to stack
        heads: Number of attention heads per layer
        concat: If True, concatenate head outputs; if False, average them
        residual: If True, use residual connections between layers
        dropout: Dropout probability for regularization
        edge_dim: Dimension of edge attributes (e.g., 1 for distance)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int=1, num_layers: int=2, heads: int=4, concat: bool=True, residual: bool=True, dropout: float=0.3, edge_dim: int = None):
        super().__init__()
        print(f"Initializing GATv2Net with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}, num_layers={num_layers}, heads={heads}, concat={concat}, residual={residual}, dropout={dropout}, edge_dim={edge_dim}")
        # Encode input features to hidden dimension
        self.node_encoder = torch.nn.Linear(input_dim, hidden_dim)
    
        # Calculate output dimension per attention head
        # Ensures even distribution of features across heads
        out_head = max(1, hidden_dim // heads)
        self.layers = torch.nn.ModuleList()

        # Calculate output dimension after each GATv2 layer
        gat_out_dim = out_head * (heads if concat else 1)
        
        # Build stack of GATv2 layers
        for i in range(num_layers):
            print(f"Adding GATv2 layer {i+1} with in_dim={hidden_dim if i == 0 else gat_out_dim}, out_dim={out_head}, heads={heads}, concat={concat}")
            # First layer takes hidden_dim; subsequent layers take previous output dimension
            in_dim = hidden_dim if i == 0 else gat_out_dim
            self.layers.append(GATv2Layer(in_dim=in_dim, out_dim=out_head, heads=heads, concat=concat, residual=residual, dropout=dropout, edge_dim=edge_dim))

        # Final classification layer
        classifier_in_dim = gat_out_dim if num_layers > 0 else hidden_dim
        self.classifier = torch.nn.Linear(classifier_in_dim, output_dim)


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> tuple[torch.Tensor, list, list]:
        """
        Forward pass through the entire GATv2 network.
        
        Args:
            x: Node feature tensor [num_nodes, input_dim]
               For epitope prediction: ESM2/ESM-IF1 embeddings of shape [num_residues, embedding_dim]
            edge_index: Graph connectivity in COO format [2, num_edges]
                       Each column represents a directed edge (source, target)
            edge_attr: Optional edge attributes [num_edges, edge_dim]
                      For protein graphs: could be 3D distances between residues
        
        Returns:
            Node-level predictions [num_nodes]
            Each value represents prediction for one node (e.g., epitope probability for one residue)
        """
        
        # Step 1: Encode input features to hidden dimension
        x = self.node_encoder(x)
        attentions = []
        edge_index_att = []
        # Step 2: Apply stacked GATv2 layers with attention mechanism
        for layer in self.layers:
            # Check edge attributes for compatibility
            if edge_attr is not None and edge_attr.size(0) != edge_index.size(1):
                print(
                    "[DEBUG] GATv2Net layer edge_attr/edge_index mismatch: "
                    f"edge_attr={tuple(edge_attr.shape)}, edge_index={tuple(edge_index.shape)}"
                )
                edge_attr = None  # Drop edge attributes if mismatched

            x, edge_index_out, attn = layer(x, edge_index, edge_attr=edge_attr)
            attentions.append(attn)  # Store attention weights for analysis
            edge_index_att.append(edge_index_out)
        # Step 3: Apply classifier to get final predictions
        # Remove last dimension for node-level binary classification
        x = self.classifier(x).squeeze(-1)
        return x, edge_index_att, attentions