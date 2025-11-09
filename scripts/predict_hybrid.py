# scripts/predict_hybrid.py
import sys, os
CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(CURRENT_FILE), ".."))
sys.path.insert(0, PROJECT_ROOT)
import argparse, pandas as pd
from src.models.hybrid_ensemble import HybridEnsemble
from src.data.data_loader import DataLoader

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--signal-col", default="vibration_rms")
    p.add_argument("--model-dir", default="models/saved_models/hybrid")
    p.add_argument("--out", default="results/predictions_hybrid.csv")
    p.add_argument("--fusion-th", type=float, default=0.6)
    args = p.parse_args()

    feats = pd.read_csv(args.features)
    raw = pd.read_csv(args.raw)

    h = HybridEnsemble.load(args.model_dir)

    if_scores = h.score_features(feats)
    lstm_scores = h.score_sequences(raw, signal_col=args.signal_col)
    fused = h.combine_scores(if_scores, lstm_scores)
    y_hat = h.decision(fused, fusion_threshold=args.fusion_th)

    # Align to available fused length
    m = len(fused)
    out = feats.head(m).copy()
    out["if_score"] = if_scores[:m]
    out["lstm_score"] = lstm_scores[:m] if len(lstm_scores) else np.zeros(m)
    out["fused_score"] = fused
    out["prediction"] = y_hat
    out.to_csv(args.out, index=False)
    print(f"✓ Saved hybrid predictions → {args.out}")

if __name__ == "__main__":
    main()
