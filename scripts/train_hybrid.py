import argparse, pandas as pd
from src.models.hybrid_ensemble import HybridEnsemble

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--signal-col", default="vibration_rms")
    p.add_argument("--save-dir", default="models/saved_models/hybrid")
    args = p.parse_args()

    feats = pd.read_csv(args.features)
    raw = pd.read_csv(args.raw)

    # ✅ FAST TRAIN MODE
    model = HybridEnsemble(
        seq_len=64,          # was 128 → faster
        lstm_units=32,       # was 64 → faster
        lstm_epochs=1,       # was 3+ → FAST
        lstm_batch=128,      # smaller batch
        fusion_weight_if=0.5,
        fusion_weight_lstm=0.5
    )

    model.fit(feats, raw, signal_col=args.signal_col)
    model.save(args.save_dir)

if __name__ == "__main__":
    main()
