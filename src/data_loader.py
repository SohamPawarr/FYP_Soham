import pandas as pd
import os

def load_elliptic():
    """Load Elliptic Data Set via KaggleHub."""
    print("\n📂 Loading Elliptic (Bitcoin)...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
        print(f"   Downloaded/Found at: {path}")

        # Find files
        edge_file = None
        class_file = None
        for root, dirs, files in os.walk(path):
            if 'elliptic_txs_edgelist.csv' in files:
                edge_file = os.path.join(root, 'elliptic_txs_edgelist.csv')
            if 'elliptic_txs_classes.csv' in files:
                class_file = os.path.join(root, 'elliptic_txs_classes.csv')
        
        if not edge_file or not class_file:
            raise FileNotFoundError("Elliptic CSVs not found in downloaded path")

        # Load
        print("   Reading CSVs...")
        edges = pd.read_csv(edge_file)
        classes = pd.read_csv(class_file)

        # Map classes: 1=Illicit, 2=Licit, "unknown"=Drop
        classes = classes[classes['class'] != 'unknown']
        classes['is_fraud'] = classes['class'].apply(lambda x: 1 if str(x) == '1' else 0)

        # Merge classes onto source node (txId1)
        df = edges.merge(classes, left_on='txId1', right_on='txId', how='inner')
        
        # Add dummy amount/timestep for flow/temporal logic
        df['amount'] = 1.0  
        df['step'] = 1.0 

        print(f"   Loaded: {len(df[df['is_fraud']==1])} illicit + {len(df[df['is_fraud']==0])} licit = {len(df)} transactions")
        return df, "txId1", "txId2", "amount", "is_fraud"

    except Exception as e:
        print(f"   ❌ Failed to load Elliptic: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), "", "", "", ""
