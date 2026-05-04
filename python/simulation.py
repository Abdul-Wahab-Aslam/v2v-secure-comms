"""
Secure Vehicle-to-Vehicle (V2V) Communication using AI
Group ID: BSIT-F25-005
Supervisor: Zunnurain Hussain

Deliverables:
  1. Traffic simulation with accident / suspicious driving scenarios
  2. AI/ML models: Isolation Forest, XGBoost, LSTM
  3. Performance metrics: accuracy, precision, recall, F1, latency
  4. JSON output for Vercel-hosted dashboard
"""

import numpy as np
import pandas as pd
import json, time, os, warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  TRAFFIC SIMULATION
# ─────────────────────────────────────────────
class V2VSimulator:
    """Simulates vehicles broadcasting telemetry in a V2V network."""

    LABELS = {0: "Normal", 1: "Accident", 2: "Suspicious"}

    def __init__(self, n_vehicles=200, timesteps=50, seed=42):
        np.random.seed(seed)
        self.n_vehicles = n_vehicles
        self.timesteps   = timesteps

    # ---------- helpers ----------
    @staticmethod
    def _normal_driving(n):
        speed       = np.random.normal(60,  10,  n).clip(0, 120)
        accel       = np.random.normal(0,    1,  n)
        braking     = np.random.normal(0.1,  0.1, n).clip(0, 1)
        lane_pos    = np.random.normal(0,    0.3, n).clip(-1, 1)
        lane_change = np.random.binomial(1,  0.05, n)
        heading_dev = np.random.normal(0,    2,  n)
        proximity   = np.random.normal(30,   5,  n).clip(5, 100)
        return speed, accel, braking, lane_pos, lane_change, heading_dev, proximity

    @staticmethod
    def _accident_driving(n):
        speed       = np.random.normal(20,  15,  n).clip(0, 80)
        accel       = np.random.normal(-8,   3,  n)
        braking     = np.random.normal(0.9,  0.05, n).clip(0, 1)
        lane_pos    = np.random.normal(0.8,  0.4, n).clip(-2, 2)
        lane_change = np.random.binomial(1,  0.6,  n)
        heading_dev = np.random.normal(30,  15,  n)
        proximity   = np.random.normal(5,    3,  n).clip(0, 20)
        return speed, accel, braking, lane_pos, lane_change, heading_dev, proximity

    @staticmethod
    def _suspicious_driving(n):
        speed       = np.random.normal(110, 15,  n).clip(60, 200)
        accel       = np.random.normal(4,    2,  n)
        braking     = np.random.normal(0.05, 0.05, n).clip(0, 0.3)
        lane_pos    = np.random.normal(0,    0.8, n).clip(-2, 2)
        lane_change = np.random.binomial(1,  0.4,  n)
        heading_dev = np.random.normal(15,  10,  n)
        proximity   = np.random.normal(8,    4,  n).clip(0, 30)
        return speed, accel, braking, lane_pos, lane_change, heading_dev, proximity

    # ---------- public ----------
    def generate(self):
        counts = {
            0: int(self.n_vehicles * 0.60),   # Normal
            1: int(self.n_vehicles * 0.20),   # Accident
            2: int(self.n_vehicles * 0.20),   # Suspicious
        }
        frames = []
        for label, n in counts.items():
            fn = [self._normal_driving,
                  self._accident_driving,
                  self._suspicious_driving][label]
            s, a, b, lp, lc, hd, pr = fn(n * self.timesteps)
            df = pd.DataFrame({
                "vehicle_id":   np.repeat(np.arange(n), self.timesteps),
                "timestep":     np.tile(np.arange(self.timesteps), n),
                "speed":        s,
                "acceleration": a,
                "braking":      b,
                "lane_position": lp,
                "lane_change":  lc,
                "heading_dev":  hd,
                "proximity":    pr,
                "label":        label,
            })
            frames.append(df)
        data = pd.concat(frames, ignore_index=True)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"[Simulation] Generated {len(data):,} records | "
              f"Normal={counts[0]*self.timesteps}, "
              f"Accident={counts[1]*self.timesteps}, "
              f"Suspicious={counts[2]*self.timesteps}")
        return data


# ─────────────────────────────────────────────
# 2.  AI PIPELINE
# ─────────────────────────────────────────────
FEATURES = ["speed", "acceleration", "braking",
            "lane_position", "lane_change", "heading_dev", "proximity"]

class ModelBenchmark:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self._prepare()

    def _prepare(self):
        X = self.data[FEATURES].values
        y = self.data["label"].values
        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # ---- Isolation Forest (unsupervised anomaly detection) ----
    def run_isolation_forest(self):
        print("\n[IsolationForest] Training …")
        t0 = time.perf_counter()
        clf = IsolationForest(n_estimators=200, contamination=0.4,
                              random_state=42, n_jobs=-1)
        clf.fit(self.X_train)
        t1 = time.perf_counter()
        # Map -1 (anomaly) → 1 (accident/suspicious), +1 (normal) → 0
        raw = clf.predict(self.X_test)
        y_pred = np.where(raw == -1, 1, 0)
        y_true_binary = np.where(self.y_test == 0, 0, 1)
        latency = round((t1 - t0) * 1000 / len(self.X_test), 3)
        return self._metrics("Isolation Forest", y_true_binary, y_pred, latency)

    # ---- XGBoost ----
    def run_xgboost(self):
        print("[XGBoost] Training …")
        t0 = time.perf_counter()
        clf = XGBClassifier(n_estimators=300, max_depth=6,
                            learning_rate=0.1, use_label_encoder=False,
                            eval_metric="mlogloss", random_state=42,
                            n_jobs=-1, verbosity=0)
        clf.fit(self.X_train, self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                verbose=False)
        t1 = time.perf_counter()
        y_pred = clf.predict(self.X_test)
        latency = round((t1 - t0) * 1000 / len(self.X_test), 3)
        return self._metrics("XGBoost", self.y_test, y_pred, latency)

    # ---- LSTM ----
    def run_lstm(self):
        print("[LSTM] Training …")
        SEQ = 10
        n_classes = 3
        # Reshape into sequences
        n_samples = len(self.X_train) - (len(self.X_train) % SEQ)
        Xtr = self.X_train[:n_samples].reshape(-1, SEQ, len(FEATURES))
        ytr = self.y_train[:n_samples].reshape(-1, SEQ)[:, -1]

        n_samples_te = len(self.X_test) - (len(self.X_test) % SEQ)
        Xte = self.X_test[:n_samples_te].reshape(-1, SEQ, len(FEATURES))
        yte = self.y_test[:n_samples_te].reshape(-1, SEQ)[:, -1]

        ytr_cat = to_categorical(ytr, num_classes=n_classes)

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQ, len(FEATURES))),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(n_classes, activation="softmax"),
        ])
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        t0 = time.perf_counter()
        model.fit(Xtr, ytr_cat, epochs=15, batch_size=64,
                  validation_split=0.1, verbose=0)
        t1 = time.perf_counter()
        y_pred = np.argmax(model.predict(Xte, verbose=0), axis=1)
        latency = round((t1 - t0) * 1000 / len(Xte), 3)
        return self._metrics("LSTM", yte, y_pred, latency)

    # ---- helpers ----
    @staticmethod
    def _metrics(name, y_true, y_pred, latency_ms):
        avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
        result = {
            "model":     name,
            "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
            "precision": round(precision_score(y_true, y_pred, average=avg,
                                               zero_division=0) * 100, 2),
            "recall":    round(recall_score(y_true, y_pred, average=avg,
                                            zero_division=0) * 100, 2),
            "f1":        round(f1_score(y_true, y_pred, average=avg,
                                        zero_division=0) * 100, 2),
            "latency_ms": latency_ms,
        }
        cm = confusion_matrix(y_true, y_pred).tolist()
        result["confusion_matrix"] = cm
        print(f"  {name}: Acc={result['accuracy']}% | "
              f"F1={result['f1']}% | Latency={latency_ms} ms/sample")
        return result


# ─────────────────────────────────────────────
# 3.  MAIN – run everything & dump JSON
# ─────────────────────────────────────────────
def main():
    sim  = V2VSimulator(n_vehicles=300, timesteps=50)
    data = sim.generate()

    # Save raw sample for inspection
    out_dir = os.path.join(os.path.dirname(__file__), "..", "web", "data")
    os.makedirs(out_dir, exist_ok=True)
    data.sample(500, random_state=1).to_csv(
        os.path.join(out_dir, "simulation_sample.csv"), index=False)

    bench   = ModelBenchmark(data)
    results = []
    results.append(bench.run_isolation_forest())
    results.append(bench.run_xgboost())
    results.append(bench.run_lstm())

    # ---- class distribution ----
    dist = data["label"].value_counts().sort_index()
    class_dist = [
        {"label": V2VSimulator.LABELS[i], "count": int(dist.get(i, 0))}
        for i in range(3)
    ]

    # ---- scenario statistics ----
    scenarios = []
    for label, gdf in data.groupby("label"):
        scenarios.append({
            "scenario": V2VSimulator.LABELS[label],
            "avg_speed":    round(gdf["speed"].mean(), 2),
            "avg_accel":    round(gdf["acceleration"].mean(), 2),
            "avg_braking":  round(gdf["braking"].mean(), 3),
            "avg_proximity":round(gdf["proximity"].mean(), 2),
        })

    # ---- per-model confusion matrices with class names ----
    for r in results:
        cm = r["confusion_matrix"]
        # Isolation Forest is binary
        if len(cm) == 2:
            r["cm_labels"] = ["Normal", "Anomaly"]
        else:
            r["cm_labels"] = ["Normal", "Accident", "Suspicious"]

    payload = {
        "meta": {
            "project": "Secure V2V Communication using AI",
            "group_id": "BSIT-F25-005",
            "supervisor": "Zunnurain Hussain",
            "members": [
                "Abdul Hanan Sabir (03-135222-001)",
                "Dania Rasool (03-135222-012)",
                "Abdul Wahab (03-135232-004)",
            ],
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "class_distribution": class_dist,
        "scenario_stats":     scenarios,
        "model_results":      results,
    }

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[Done] Results saved → {out_path}")


if __name__ == "__main__":
    main()
