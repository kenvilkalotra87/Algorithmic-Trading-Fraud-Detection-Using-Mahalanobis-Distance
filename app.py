"""
TRADEGUARD - Algorithmic Trading Anomaly Detection Application.
Course: Pattern Recognition and Anomaly Detection (Semester 5)

Official Problem Statement:
'Build a Mahalanobis distance pipeline to flag irregular high-frequency trades
and volume spikes in simulated financial markets.'

Backend Server built with Flask.
"""

import os
import io
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
from ml.mahalanobis_detector import MahalanobisTradingDetector

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload limit
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "data")
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

DEFAULT_DATASET_PATH = os.path.join(UPLOAD_FOLDER, "trading_data.csv")
RESULTS_PATH = os.path.join(RESULTS_FOLDER, "detected_anomalies.csv")

# Global state for active dataset and detector
class AppState:
    def __init__(self):
        self.current_dataset_path = DEFAULT_DATASET_PATH
        self.detector = MahalanobisTradingDetector(mahalanobis_threshold=3.0, volume_z_threshold=3.0)
        self.raw_df = None
        self.scored_df = None
        self.summary_stats = None
        self.initialize_state()

    def initialize_state(self):
        if os.path.exists(self.current_dataset_path):
            try:
                self.raw_df = pd.read_csv(self.current_dataset_path)
                self.run_detection()
            except Exception as e:
                print(f"Error initializing dataset: {e}")

    def run_detection(self, mahalanobis_threshold=None, volume_z_threshold=None):
        if mahalanobis_threshold is not None:
            self.detector.mahalanobis_threshold = float(mahalanobis_threshold)
        if volume_z_threshold is not None:
            self.detector.volume_z_threshold = float(volume_z_threshold)

        # Fit detector on raw data and score
        self.detector.fit(self.raw_df)
        self.scored_df = self.detector.detect(self.raw_df)
        self.summary_stats = self.detector.get_summary_statistics(self.scored_df)

        # Save detected anomalies to CSV
        anomalies_only = self.scored_df[self.scored_df["is_anomaly"] == True]
        anomalies_only.to_csv(RESULTS_PATH, index=False)
        return self.summary_stats

state = AppState()

# ---------------------------------------------------------
# Web Page Route
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ---------------------------------------------------------
# REST APIs
# ---------------------------------------------------------
@app.route("/api/dashboard", methods=["GET"])
def get_dashboard():
    """
    Returns dynamically computed dashboard statistics and configuration.
    """
    if state.summary_stats is None:
        state.run_detection()

    return jsonify({
        "status": "success",
        "dataset_name": os.path.basename(state.current_dataset_path),
        "statistics": state.summary_stats
    })

@app.route("/api/market-data", methods=["GET"])
def get_market_data():
    """
    Returns structured data for Chart.js visualizations:
    - Normal vs Anomaly Doughnut
    - Volume distribution histogram
    - Mahalanobis distance distribution histogram
    - Price vs Volume scatter (downsampled for smooth UI rendering)
    - Price vs Quantity scatter
    - Volume vs Price Change scatter
    - Price movement line chart
    - Trading activity over time
    """
    if state.scored_df is None:
        state.run_detection()

    df = state.scored_df

    # 1. Doughnut Chart data
    normal_count = int(state.summary_stats["normal_trades"])
    anomaly_count = int(state.summary_stats["anomalous_trades"])

    # 2. Mahalanobis Distance Histogram (25 bins)
    md_values = df["mahalanobis_distance"].values
    md_max = min(float(np.percentile(md_values, 99.5) * 1.2), float(np.max(md_values)))
    md_bins = np.linspace(0, max(md_max, 6.0), 25)
    md_counts, md_edges = np.histogram(md_values, bins=md_bins)
    md_labels = [f"{round(md_edges[i], 1)}-{round(md_edges[i+1], 1)}" for i in range(len(md_edges)-1)]

    # 3. Volume Distribution Histogram (25 bins)
    vol_values = df["volume"].values
    vol_max = min(float(np.percentile(vol_values, 99.2) * 1.2), float(np.max(vol_values)))
    vol_bins = np.linspace(0, max(vol_max, 50000.0), 25)
    vol_counts, vol_edges = np.histogram(vol_values, bins=vol_bins)
    vol_labels = [f"${int(vol_edges[i]/1000)}k-${int(vol_edges[i+1]/1000)}k" for i in range(len(vol_edges)-1)]

    # 4. Downsampled Scatter Plot Data (to ensure fast, responsive 60fps rendering in browser)
    # Include all anomalies, and downsample normal trades to 700 points
    anomalies_df = df[df["is_anomaly"] == True]
    normals_df = df[df["is_anomaly"] == False]
    normal_sample = normals_df.sample(min(700, len(normals_df)), random_state=42)

    # Price vs Volume
    pv_normal = [{"x": round(float(r["price"]), 2), "y": round(float(r["volume"]), 2), "id": str(r["trade_id"]), "sym": str(r["symbol"])} for _, r in normal_sample.iterrows()]
    pv_anomaly = [{"x": round(float(r["price"]), 2), "y": round(float(r["volume"]), 2), "id": str(r["trade_id"]), "sym": str(r["symbol"]), "reason": str(r["reason"]), "md": round(float(r["mahalanobis_distance"]), 2)} for _, r in anomalies_df.iterrows()]

    # Price vs Quantity
    pq_normal = [{"x": round(float(r["price"]), 2), "y": int(r["quantity"]), "id": str(r["trade_id"]), "sym": str(r["symbol"])} for _, r in normal_sample.iterrows()]
    pq_anomaly = [{"x": round(float(r["price"]), 2), "y": int(r["quantity"]), "id": str(r["trade_id"]), "sym": str(r["symbol"]), "reason": str(r["reason"]), "md": round(float(r["mahalanobis_distance"]), 2)} for _, r in anomalies_df.iterrows()]

    # Trade Volume vs Price Change
    vp_normal = [{"x": round(float(r["price_change"]), 3), "y": round(float(r["volume"]), 2), "id": str(r["trade_id"]), "sym": str(r["symbol"])} for _, r in normal_sample.iterrows()]
    vp_anomaly = [{"x": round(float(r["price_change"]), 3), "y": round(float(r["volume"]), 2), "id": str(r["trade_id"]), "sym": str(r["symbol"]), "reason": str(r["reason"]), "md": round(float(r["mahalanobis_distance"]), 2)} for _, r in anomalies_df.iterrows()]

    # 5. Price Movement Line Chart (Time-series sample of primary symbols)
    # Take a chronological slice of ~120 timestamp points
    ts_sample = df.iloc[::max(1, len(df) // 100)].copy()
    time_labels = [str(ts).split(" ")[-1][:8] for ts in ts_sample["timestamp"]]
    price_series = {
        sym: [round(float(r["price"]), 2) if r["symbol"] == sym else None for _, r in ts_sample.iterrows()]
        for sym in ["TECH", "BANK", "AUTO", "ENERGY", "PHARMA"]
    }

    # 6. Trading Activity Over Time (Binned trade counts across the trading day)
    # Group by 15-minute or chronological buckets
    bucket_size = max(1, len(df) // 20)
    activity_labels = []
    activity_total = []
    activity_anomalies = []
    for chunk_start in range(0, len(df), bucket_size):
        chunk = df.iloc[chunk_start:chunk_start + bucket_size]
        start_time_str = str(chunk.iloc[0]["timestamp"]).split(" ")[-1][:5]
        activity_labels.append(start_time_str)
        activity_total.append(int(len(chunk)))
        activity_anomalies.append(int(chunk["is_anomaly"].sum()))

    return jsonify({
        "status": "success",
        "doughnut": {
            "labels": ["Normal Trades", "Potential Anomalies"],
            "data": [normal_count, anomaly_count]
        },
        "md_histogram": {
            "labels": md_labels,
            "counts": [int(c) for c in md_counts],
            "threshold": state.detector.mahalanobis_threshold
        },
        "volume_histogram": {
            "labels": vol_labels,
            "counts": [int(c) for c in vol_counts],
            "mean_volume": state.summary_stats["average_volume"]
        },
        "scatter": {
            "price_volume": {"normal": pv_normal, "anomaly": pv_anomaly},
            "price_quantity": {"normal": pq_normal, "anomaly": pq_anomaly},
            "volume_price_change": {"normal": vp_normal, "anomaly": vp_anomaly}
        },
        "price_movement": {
            "labels": time_labels,
            "series": price_series
        },
        "activity_time": {
            "labels": activity_labels,
            "total_trades": activity_total,
            "anomaly_trades": activity_anomalies
        }
    })

@app.route("/api/trades", methods=["GET"])
def get_trades():
    """
    Returns paginated and filtered trade records.
    Query parameters:
    - page (int, default 1)
    - page_size (int, default 20)
    - search (str, trade_id prefix or match)
    - symbol (str, symbol code or 'ALL')
    - status (str, 'ALL', 'POTENTIAL ANOMALY', 'NORMAL')
    - reason (str, 'ALL', 'Mahalanobis Distance', 'Volume Spike', 'Both')
    - sort_by (str, column name)
    - sort_order (str, 'asc' or 'desc')
    """
    if state.scored_df is None:
        state.run_detection()

    df = state.scored_df

    # Search filter
    search = request.args.get("search", "").strip()
    if search:
        df = df[df["trade_id"].str.contains(search, case=False, na=False)]

    # Symbol filter
    symbol = request.args.get("symbol", "ALL").strip().upper()
    if symbol and symbol != "ALL":
        df = df[df["symbol"] == symbol]

    # Status filter
    status = request.args.get("status", "ALL").strip().upper()
    if status and status != "ALL":
        df = df[df["status"] == status]

    # Reason filter
    reason = request.args.get("reason", "ALL").strip()
    if reason and reason != "ALL":
        df = df[df["reason"] == reason]

    # Sorting
    sort_by = request.args.get("sort_by", "mahalanobis_distance")
    sort_order = request.args.get("sort_order", "desc").lower()
    ascending = (sort_order == "asc")

    allowed_sort_columns = [
        "trade_id", "timestamp", "symbol", "price",
        "quantity", "volume", "price_change", "mahalanobis_distance", "volume_zscore"
    ]
    if sort_by in allowed_sort_columns and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    total_filtered = len(df)

    # Pagination
    try:
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(1, int(request.args.get("page_size", 20))))
    except ValueError:
        page = 1
        page_size = 20

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    total_pages = max(1, int(np.ceil(total_filtered / page_size)))

    page_rows = df.iloc[start_idx:end_idx].to_dict(orient="records")

    return jsonify({
        "status": "success",
        "page": page,
        "page_size": page_size,
        "total_records": total_filtered,
        "total_pages": total_pages,
        "trades": page_rows
    })

@app.route("/api/trade/<trade_id>", methods=["GET"])
def get_trade_detail(trade_id):
    """
    Returns complete analytical profile and diagnostic breakdown for a single trade.
    """
    if state.scored_df is None:
        state.run_detection()

    match = state.scored_df[state.scored_df["trade_id"] == trade_id.strip()]
    if match.empty:
        return jsonify({"status": "error", "message": f"Trade ID '{trade_id}' not found."}), 404

    row = match.iloc[0].to_dict()

    # Calculate comparison with symbol averages
    sym = row["symbol"]
    sym_group = state.scored_df[state.scored_df["symbol"] == sym]
    sym_avg_price = float(sym_group["price"].mean())
    sym_avg_volume = float(sym_group["volume"].mean())
    sym_avg_qty = float(sym_group["quantity"].mean())

    comparison = {
        "symbol_avg_price": round(sym_avg_price, 2),
        "symbol_avg_volume": round(sym_avg_volume, 2),
        "symbol_avg_quantity": round(sym_avg_qty, 1),
        "price_diff_pct": round(((row["price"] - sym_avg_price) / sym_avg_price) * 100, 2),
        "volume_diff_pct": round(((row["volume"] - sym_avg_volume) / sym_avg_volume) * 100, 2),
        "market_mean_vector": {
            feat: round(float(state.detector.feature_means.get(feat, 0.0)), 3)
            for feat in state.detector.FEATURE_COLUMNS
        }
    }

    return jsonify({
        "status": "success",
        "trade": row,
        "comparison": comparison,
        "thresholds": {
            "mahalanobis_threshold": state.detector.mahalanobis_threshold,
            "volume_z_threshold": state.detector.volume_z_threshold
        }
    })

@app.route("/api/detect", methods=["POST"])
def detect_anomalies():
    """
    Re-runs detection with user-specified thresholds and optional CSV upload.
    """
    try:
        md_thresh = request.form.get("mahalanobis_threshold")
        vol_thresh = request.form.get("volume_z_threshold")

        # Check if file was uploaded in this request
        if "file" in request.files:
            file = request.files["file"]
            if file and file.filename != "":
                if not file.filename.endswith(".csv"):
                    return jsonify({"status": "error", "message": "Only CSV files are supported."}), 400

                # Validate CSV content
                try:
                    uploaded_df = pd.read_csv(file)
                except Exception as e:
                    return jsonify({"status": "error", "message": f"Failed to parse CSV: {str(e)}"}), 400

                valid, msg = state.detector.validate_data(uploaded_df)
                if not valid:
                    return jsonify({"status": "error", "message": msg}), 400

                # Save file
                filename = secure_filename(file.filename)
                save_path = os.path.join(UPLOAD_FOLDER, f"upload_{filename}")
                uploaded_df.to_csv(save_path, index=False)
                state.current_dataset_path = save_path
                state.raw_df = uploaded_df

        # If thresholds were sent as JSON payload instead of FormData
        if request.is_json:
            data = request.get_json()
            md_thresh = data.get("mahalanobis_threshold", md_thresh)
            vol_thresh = data.get("volume_z_threshold", vol_thresh)

        # Parse threshold values
        md_val = float(md_thresh) if md_thresh is not None else state.detector.mahalanobis_threshold
        vol_val = float(vol_thresh) if vol_thresh is not None else state.detector.volume_z_threshold

        if md_val <= 0 or vol_val <= 0:
            return jsonify({"status": "error", "message": "Thresholds must be positive numbers greater than 0."}), 400

        # Execute detection pipeline
        summary = state.run_detection(mahalanobis_threshold=md_val, volume_z_threshold=vol_val)

        # Retrieve top 10 anomalous trades for immediate feedback
        top_anomalies = state.scored_df[state.scored_df["is_anomaly"] == True].sort_values(
            by="mahalanobis_distance", ascending=False
        ).head(10).to_dict(orient="records")

        return jsonify({
            "status": "success",
            "message": "Detection completed successfully.",
            "dataset_name": os.path.basename(state.current_dataset_path),
            "statistics": summary,
            "top_anomalies": top_anomalies
        })

    except ValueError as ve:
        return jsonify({"status": "error", "message": f"Invalid numerical parameter: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Detection error: {str(e)}"}), 500

@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    """
    Dedicated CSV dataset upload route with comprehensive validation.
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected."}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"status": "error", "message": "Invalid file format. Please upload a .csv file."}), 400

    try:
        df = pd.read_csv(file)
        valid, msg = state.detector.validate_data(df)
        if not valid:
            return jsonify({"status": "error", "message": msg}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        df.to_csv(save_path, index=False)

        state.current_dataset_path = save_path
        state.raw_df = df
        summary = state.run_detection()

        return jsonify({
            "status": "success",
            "message": f"Successfully uploaded and analyzed '{filename}'.",
            "dataset_name": filename,
            "statistics": summary
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error processing file: {str(e)}"}), 400

@app.route("/api/results", methods=["GET"])
def get_results_summary():
    """
    Returns results overview and top flagged records for the Results & Trades page.
    """
    if state.scored_df is None:
        state.run_detection()

    df = state.scored_df
    anomalies_df = df[df["is_anomaly"] == True]

    top_md = anomalies_df.sort_values(by="mahalanobis_distance", ascending=False).head(15).to_dict(orient="records")
    top_vol = anomalies_df.sort_values(by="volume_zscore", ascending=False).head(15).to_dict(orient="records")

    return jsonify({
        "status": "success",
        "dataset_name": os.path.basename(state.current_dataset_path),
        "statistics": state.summary_stats,
        "top_by_mahalanobis": top_md,
        "top_by_volume": top_vol
    })

@app.route("/api/download-results", methods=["GET"])
def download_results():
    """
    Exports detected anomalies as a downloadable CSV.
    """
    if not os.path.exists(RESULTS_PATH) or state.scored_df is None:
        state.run_detection()

    return send_file(
        RESULTS_PATH,
        mimetype="text/csv",
        as_attachment=True,
        download_name="trade_anomaly_detection_results.csv"
    )

@app.route("/api/reset-default-dataset", methods=["POST"])
def reset_default_dataset():
    """
    Resets the active dataset back to the default synthetic HFT market dataset.
    """
    state.current_dataset_path = DEFAULT_DATASET_PATH
    state.initialize_state()
    return jsonify({
        "status": "success",
        "message": "Reset to default synthetic market dataset.",
        "dataset_name": os.path.basename(DEFAULT_DATASET_PATH),
        "statistics": state.summary_stats
    })

if __name__ == "__main__":
    print("Starting TRADEGUARD Flask Server on http://127.0.0.1:5001 ...")
    app.run(host="127.0.0.1", port=5001, debug=True)
