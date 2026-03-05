"""Multi-view Dataset for MGAT.

Combines multiple graph views for the same samples into a single PyG `Data`.

Important:
- All views must refer to the *same node set* (same ordering and same `num_nodes`).
- When batching with PyG, custom attributes like `edge_index_0` must be incremented
  by `num_nodes` between samples. We implement this via a small `Data` subclass.
"""

import torch
from torch_geometric.data import Data, Dataset


class MultiViewData(Data):
    """PyG Data that properly batches `edge_index_{k}` attributes and all node/edge attributes."""

    def __inc__(self, key, value, *args, **kwargs):
        """Increment indices during batching."""
        if key.startswith('edge_index_'):
            return int(self.num_nodes)
        if key == 'edge_index':
            return int(self.num_nodes)
        # Don't increment these attributes
        if key in {'name', 'node_id'}:
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        """Define concatenation dimension for each attribute during batching."""
        # Edge indices: concatenate along edge dimension (dimension 1)
        if key.startswith('edge_index_'):
            return 1
        if key == 'edge_index':
            return 1
        
        # Node-level attributes: concatenate along node dimension (dimension 0)
        if key in {'x', 'node_attrs', 'coords', 'rsa', 'train_mask', 'y'}:
            return 0
        
        # Edge-level attributes
        if key == 'edge_attr':
            return 0  # Concatenate along edge dimension (assuming [num_edges, features])
        
        # Metadata: don't concatenate, just keep first value or aggregate properly
        if key in {'name', 'node_id', 'num_nodes', 'batch', 'ptr'}:
            return None
        
        return super().__cat_dim__(key, value, *args, **kwargs)
    
    def get_all_keys(self):
        """Return all keys in this data object for easy inspection."""
        return list(self.keys())
    
    def get_node_features(self):
        """Easy access to node features (tries 'x' first, then 'node_attrs')."""
        if 'x' in self:
            return self['x']
        elif 'node_attrs' in self:
            return self['node_attrs']
        else:
            raise KeyError("Neither 'x' nor 'node_attrs' found in data object")



class MultiViewDataset(Dataset):
    """
    Dataset that combines multiple views of the same graphs.
    Each view should have the same number of samples and correspond to the same entities.
    """
    
    def __init__(self, view_datasets):
        """
        Args:
            view_datasets (list of lists): List of K lists, where each list contains graph data for one view.
                                          All lists should have the same length N (number of samples).
        """
        super().__init__()
        self.view_datasets = view_datasets
        self.num_views = len(view_datasets)
        self.num_samples = len(view_datasets[0])
        
        # Verify all views have the same number of samples
        for i, view in enumerate(view_datasets):
            if len(view) != self.num_samples:
                raise ValueError(f"View {i} has {len(view)} samples, but view 0 has {self.num_samples} samples")
    
    def len(self):
        return self.num_samples
    
    def get(self, idx):
        """
        Get the multi-view graph at index idx.
        Returns a Data object with edge_index_0, edge_index_1, etc. for each view.
        """
        # Get the first view as the base
        base_graph = self.view_datasets[0][idx]
        base_num_nodes = int(base_graph.num_nodes)
        
        # Create a dictionary for the multi-view data
        data_dict = {
            'num_nodes': base_num_nodes,
        }
        
        # Use node_attrs as x if available, otherwise use x
        if 'node_attrs' in base_graph:
            node_features = base_graph['node_attrs']
        elif 'x' in base_graph:
            node_features = base_graph['x']
        else:
            raise ValueError(
                f"MultiViewDataset sample idx={idx} has neither 'node_attrs' nor 'x'. "
                f"Available keys: {list(base_graph.keys())}"
            )
        
        # Store node features in both x (for PyG batching) and node_attrs (for model)
        data_dict['x'] = node_features
        data_dict['node_attrs'] = node_features
        
        # Add labels
        if 'y' in base_graph:
            data_dict['y'] = base_graph['y']
        
        # Add edge indices from all views with numbered keys
        for i, view in enumerate(self.view_datasets):
            view_graph = view[idx]
            view_num_nodes = int(view_graph.num_nodes)
            if view_num_nodes != base_num_nodes:
                base_id = None
                for cand in ('id', 'name', 'pdb_id', 'pdb', 'protein_id', 'chain_id'):
                    if cand in base_graph:
                        base_id = base_graph[cand]
                        break
                raise ValueError(
                    f"MultiViewDataset sample idx={idx} has inconsistent num_nodes across views: "
                    f"view0={base_num_nodes}, view{i}={view_num_nodes}. "
                    f"(sample_id={base_id})"
                )

            edge_index = view_graph.edge_index
            if edge_index is None:
                raise ValueError(f"MultiViewDataset sample idx={idx} view{i} has no edge_index")
            if edge_index.dtype != torch.long:
                edge_index = edge_index.long()
            data_dict[f'edge_index_{i}'] = edge_index
        
        # Copy any additional attributes from the base graph (excluding edge_index which is per-view)
        for key in base_graph.keys():
            if key not in ['x', 'node_attrs', 'y', 'edge_index', 'num_nodes']:
                data_dict[key] = base_graph[key]
        
        # Create a new Data object
        multi_view_data = MultiViewData(**data_dict)
        
        return multi_view_data
