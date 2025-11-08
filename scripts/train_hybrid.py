# scripts/train_hybrid.py
import argparse, pandas as pd
from src.models.hybrid_ensemble import HybridEnsemble

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="Path to features CSV (e.g., data/features/machine_001_features.csv)")
    p.add_argument("--raw", required=True, help="Path to raw CSV (e.g., data/synthetic/machine_001_data.csv)")
    p.add_argument("--signal-col", default="vibration_rms")
    p.add_argument("--save-dir", default="models/saved_models/hybrid")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lstm-units", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()

    feats = pd.read_csv(args.features)
    raw = pd.read_csv(args.raw)

    model = HybridEnsemble(seq_len=args.seq_len,
                           lstm_units=args.lstm_units,
                           lstm_epochs=args.epochs,
                           lstm_batch=args.batch,
                           fusion_weight_if=0.5,
                           fusion_weight_lstm=0.5)
    model.fit(feats, raw, signal_col=args.signal_col)
    model.save(args.save_dir)

if __name__ == "__main__":
    main()
