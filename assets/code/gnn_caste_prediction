!pip install torch-geometric

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# --- PyTorch Geometric Imports ---
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import dense_to_sparse
import os
import scipy.io
import scipy.sparse
from imblearn.pipeline import Pipeline

NODE_FEATURES_FILE = 'new_hh_data_c.csv' # Your CSV with all features/labels
ADJ_MATRIX_FILE = 'adjacencymatrix.mat' # Your .mat file
DATA_DIR = '.' # Directory where files are located

# --- Village Selection ---

VILLAGE_NUMBERS = np.array([
    1, 2, 3, 4, 6, 9, 12, 15, 19, 20, 21, 23, 24, 25, 29, 31, 32, 33, 36,
    39, 42, 45, 46, 47, 48, 50, 51, 52, 55, 59, 60, 62, 64, 65, 67, 68,
    70, 71, 72, 73, 75
])

VILLAGE_INDICES = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                            17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42])


# --- Column Names ---
TARGET_COLUMN = 'castesubcaste'
VILLAGE_ID_COLUMN = 'village'

numeric_features = ["room_no", "bed_no", "hh_size", "women", "min_age", "max_age", "deg_cnt", "clust", 'eigen', 'between',
                    'closeness_freeman', 'information_cen', 'local_bridging']
categorical_features = ['rooftype1', 'rooftype2','rooftype3','rooftype4', 'rooftype5', 'ownrent', 'rel_num',
                        'elect_num', 'latrine_num', 'leader', 'village']
VAL_SET_SIZE = 0.15

def load_all_graph_data():
    """
    Loads data from .mat and .csv, preprocesses, and builds a list of
    Data objects (one for each village).
    """

    # --- 1. Load Adjacency Matrices ---
    print(f"Loading adjacency matrices from {ADJ_MATRIX_FILE}...")
    try:
        X_data = scipy.io.loadmat(os.path.join(DATA_DIR, ADJ_MATRIX_FILE))
        X_graphs_all = X_data['X'].ravel()
    except Exception as e:
        print(f"Error loading {ADJ_MATRIX_FILE}: {e}")
        return None

    try:
        adj_matrices = X_graphs_all[VILLAGE_INDICES]
        assert len(adj_matrices) == len(VILLAGE_NUMBERS), \
            "Number of matrices and village numbers does not match"
    except Exception as e:
        print(f"Error filtering adjacency matrices by indices: {e}")
        return None

    # --- 2. Load Node Features ---
    print(f"Loading node features from {NODE_FEATURES_FILE}...")
    try:
        all_node_data = pd.read_csv(NODE_FEATURES_FILE)
    except FileNotFoundError:
        print(f"ERROR: Could not find file {NODE_FEATURES_FILE}")
        return None

    # --- 3. Preprocess Labels (y) ---
    print("Preprocessing labels...")
    all_node_data['__original_caste__'] = all_node_data[TARGET_COLUMN]

    # Drop 'MINORITY' by replacing it with NaN
    # This treats it as a missing value to be predicted
    all_node_data[TARGET_COLUMN] = all_node_data[TARGET_COLUMN].replace('MINORITY', np.nan)

    # Use LabelEncoder on *all* non-NA values to get 0-N mapping
    label_encoder = LabelEncoder()
    known_labels = all_node_data[all_node_data[TARGET_COLUMN].notna()][TARGET_COLUMN]
    label_encoder.fit(known_labels)

    # Map labels to integers, keep NAs as -1
    all_node_data[TARGET_COLUMN] = all_node_data[TARGET_COLUMN].map(
        lambda x: label_encoder.transform([x])[0] if pd.notna(x) else -1
    )

    num_classes = len(label_encoder.classes_)
    print(f"Found {num_classes} classes: {label_encoder.classes_}")

    # --- 4. Preprocess Features (X) ---
    print("Preprocessing all features...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    feature_cols = numeric_features + categorical_features

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Fit and transform all features at once
    all_features_processed = preprocessor.fit_transform(all_node_data[feature_cols])

    try:
        all_features_processed = all_features_processed.toarray()
    except AttributeError:
        pass

    num_features = all_features_processed.shape[1]
    print(f"Processed data to {num_features} features.")

    # --- 5. Build Graph List ---
    graph_list = []

    original_index_map_list = [] # To store the original DataFrame indices

    print(f"Building {len(VILLAGE_NUMBERS)} graphs...")

    for i, village_num in enumerate(VILLAGE_NUMBERS):
        village_mask = all_node_data[VILLAGE_ID_COLUMN] == village_num
        village_nodes = all_node_data[village_mask]

        # Store the original DataFrame indices for this village's nodes
        original_index_map_list.append(village_nodes.index.values)

        if len(village_nodes) == 0:
            print(f"Warning: No nodes found for village {village_num}. Skipping.")
            continue

        village_features_processed = all_features_processed[village_nodes.index]

        # --- Get Features (x) ---
        x = torch.tensor(village_features_processed, dtype=torch.float)

        # --- Get Labels (y) ---
        y = torch.tensor(village_nodes[TARGET_COLUMN].values, dtype=torch.long)

        # --- Get Adjacency (edge_index) ---
        adj_matrix_obj = adj_matrices[i]

        if scipy.sparse.issparse(adj_matrix_obj):
            adj_matrix = adj_matrix_obj.toarray() # Convert to dense
        else:
            adj_matrix = adj_matrix_obj

        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)
        edge_index, _ = dense_to_sparse(adj_tensor)

        # --- Create Masks (Train/Val/Predict) ---
        num_nodes_in_village = len(village_nodes)

        known_indices = (y != -1).nonzero(as_tuple=False).view(-1)
        missing_indices = (y == -1).nonzero(as_tuple=False).view(-1)

        if len(known_indices) > 0:
            # Split *known* indices into train and validation
            train_indices, val_indices = train_test_split(
                known_indices, test_size=VAL_SET_SIZE, random_state=42
            )
        else:
            train_indices = torch.tensor([], dtype=torch.long)
            val_indices = torch.tensor([], dtype=torch.long)

        train_mask = torch.zeros(num_nodes_in_village, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes_in_village, dtype=torch.bool)
        predict_mask = torch.zeros(num_nodes_in_village, dtype=torch.bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        predict_mask[missing_indices] = True

        # Set missing labels (y == -1) to 0
        y[y == -1] = 0

        graph_list.append(Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            predict_mask=predict_mask,
            village_num=village_num
        ))
        original_index_map = np.concatenate(original_index_map_list)

    print(f"Successfully built {len(graph_list)} graphs.")
    return graph_list, num_features, num_classes, label_encoder, all_node_data, original_index_map


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        self.dropout = torch.nn.Dropout(p=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

def main():
    # 1. Load Data
    load_result = load_all_graph_data()
    if load_result is None:
        print("Stopping due to data loading error.")
        return

    graph_list, num_features, num_classes, label_encoder, all_node_data, original_index_map = load_result

    # DataLoader combines all graphs into one giant disconnected graph
    # This is highly efficient.
    loader = DataLoader(graph_list, batch_size=len(graph_list), shuffle=False)
    data = next(iter(loader)) # Get the one giant batched graph

    # 2. Initialize Model, Optimizer, Loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    hidden_dim = 100
   # num_heads = 5 # for GAT with attention layer

    model = SAGE(in_channels=num_features,
            hidden_channels=hidden_dim,
            out_channels=num_classes).to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00003, weight_decay=6e-5)
    criterion = torch.nn.NLLLoss()\

    print("--- Starting GNN Training ---")
    # 3. Training Loop
    for epoch in range(1, 25000):
        model.train()
        optimizer.zero_grad()

        out = model(data) # Pass the *entire* batched graph

        # Loss is calculated *only* on the nodes in the train_mask
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data)
                val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
                pred = val_out.argmax(dim=1)

                # Check accuracy on training nodes
                train_correct = pred[data.train_mask] == data.y[data.train_mask]
                train_acc = int(train_correct.sum()) / int(data.train_mask.sum())

                # Check accuracy on validation nodes
                val_correct = pred[data.val_mask] == data.y[data.val_mask]
                val_acc = int(val_correct.sum()) / int(data.val_mask.sum())

            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # 4. Final Prediction
    print("Training finished. Predicting missing values...")
    model.eval()
    with torch.no_grad():
        final_out = model(data)
        final_predictions = final_out.argmax(dim=1)

        # Get predictions ONLY for the nodes in the predict_mask
        pyg_indices_to_predict = final_predictions[data.predict_mask].cpu().numpy()

        # Get the *original DataFrame indices* of these nodes
        original_pyg_indices_to_predict = data.predict_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()

    # 5. Map Predictions Back to DataFrame

    # Convert predicted class indices (0-4) back to labels ('Gen', 'OBC'...)
    predicted_labels = label_encoder.inverse_transform(pyg_indices_to_predict)

    # Initialize the new column as 'object' dtype to hold strings
    all_node_data['predicted_caste_gnn'] = pd.Series(dtype='object')


    # 1. Use our map to get the *actual DataFrame index labels that correspond to the missing nodes.
    df_index_labels_to_fill = original_index_map[original_pyg_indices_to_predict]

    # 2. Use .loc to assign the labels to the correct rows in the new column
    all_node_data.loc[df_index_labels_to_fill, 'predicted_caste_gnn'] = predicted_labels

    # 3. Restore the original caste column
    all_node_data[TARGET_COLUMN] = all_node_data['__original_caste__']

    # Use combine_first (or a non-inplace .fillna) to fill NAs
    # This is safer and avoids the warning.
    all_node_data[TARGET_COLUMN] = all_node_data[TARGET_COLUMN].fillna(
        all_node_data['predicted_caste_gnn']
    )

    # Clean up helper columns
    all_node_data.drop(columns=['__original_caste__', 'predicted_caste_gnn'], inplace=True)

    print("\n--- Imputation Complete ---")
    print("\nFinal caste counts (after GNN imputation):")
    print(all_node_data[TARGET_COLUMN].value_counts(dropna=False))

    # Save the final results
    all_node_data.to_csv("nodes_with_gnn_predictions.csv", index=False)
    print("\nSaved final data to 'nodes_with_gnn_predictions.csv'")

    print("\n--- Final Model Evaluation on Validation Set ---")
    model.eval()
    with torch.no_grad():
        final_out = model(data.to(device))
        pred = final_out.argmax(dim=1)

        # Get the true labels and predictions *only* for the validation set
        y_true_val = data.y[data.val_mask].cpu().numpy()
        y_pred_val = pred[data.val_mask].cpu().numpy()

        # Get the class names from the encoder
        class_names = label_encoder.classes_

        # Print the report
        from sklearn.metrics import classification_report
        print(classification_report(y_true_val, y_pred_val, target_names=class_names))

if __name__ == '__main__':
    main()
