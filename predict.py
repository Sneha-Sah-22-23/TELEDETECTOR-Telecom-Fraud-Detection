import mlflow.sklearn
import pandas as pd
from collections import deque
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 1. Load the latest model
model = mlflow.sklearn.load_model("models:/TelecomFraudDetector/9")

# 2. Initialize the Priority Investigation Queue
investigation_queue = deque()

def main():
    # Load unseen test data and sim registry
    try:
        test_df = pd.read_csv("data/unseen_test_data.csv")
        sim_data = pd.read_csv("data/sim_registrations.csv")
    except FileNotFoundError:
        print("❌ Error: Files not found. Ensure train.py has run.")
        return

    print(f"🚨 Starting Batch Analysis on {len(test_df)} unseen records...")

    merged_data = test_df.merge(sim_data[['phone_number', 'owner_id']], on='phone_number', how='left')
    
    # Count SIMs per owner for the whole registry once
    sim_counts = sim_data['owner_id'].value_counts().to_dict()

    flagged_count = 0
    total_records = len(merged_data)

    # Drop non-feature columns
    feature_cols = [c for c in merged_data.columns if c not in ['is_fraud', 'phone_number', 'owner_id']]
    features = merged_data[feature_cols].values
    
    # Predict all at once instead of row by row
    ml_predictions = model.predict(features)
    
    # Layer 2 — SIM farm check all at once
    merged_data['sim_count'] = merged_data['owner_id'].map(sim_counts).fillna(0)
    sim_farm_triggers = (merged_data['sim_count'] >= 10).astype(int).values
    
    # Final decision
    flagged_mask = (ml_predictions == 1) | (sim_farm_triggers == 1)
    flagged_numbers = merged_data.loc[flagged_mask, 'phone_number'].tolist()
    investigation_queue.extend(flagged_numbers)
    flagged_count = len(flagged_numbers)

    # ── FINAL REPORT ───
    print("\n" + "─"*40)
    print("      BATCH INVESTIGATION REPORT")
    print("="*40)
    print(f"Total Numbers Scanned   : {total_records}")
    print(f"Numbers Flagged (Fraud) : {flagged_count}")
    print(f"Legitimate Numbers      : {total_records - flagged_count}")
    print(f"Queue Size              : {len(investigation_queue)}")
    print("─"*40)
    
    if investigation_queue:
        print(f"\nFirst 5 numbers in Priority Queue: {list(investigation_queue)[:5]}")
    print("Analysis Complete. All flagged numbers added to investigation_queue.")

if __name__ == "__main__":
    main()