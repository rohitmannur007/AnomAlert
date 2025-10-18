import os
import yaml
import joblib
import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

from src.models import load_models, anomaly_score_autoencoder, anomaly_score_isoforest
from src.detect import load_preprocessing_tools, prepare_sample, detect_anomaly
from src.data import load_raw, preprocess, ensure_dirs

# load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

ensure_dirs(cfg)

MODEL_DIR = cfg['models']['path']

# load preprocessing tools & models
encoders, scaler = load_preprocessing_tools(MODEL_DIR)
models = load_models(MODEL_DIR)

def inspect_data(n=5):
    df = load_raw(cfg)
    df_proc, _, _ = preprocess(df, cfg, save_encoders=False)
    return df_proc.head(n).reset_index().to_dict(orient='records')

def run_detection(sample_dict):
    # sample_dict is a mapping of feature->value (strings/numbers)
    sample_df = pd.DataFrame([sample_dict])
    # make sure columns same order/types as training; preprocess uses encoders/scaler
    result = detect_anomaly(sample_df, models, encoders, scaler)
    return result

def show_scores_time_series(periods=120):
    # Build aggregated anomaly score time-series from raw data (simple demo)
    df = load_raw(cfg)
    df_proc, encoders_local, scaler_local = preprocess(df, cfg, save_encoders=False)
    X = df_proc.select_dtypes(include=['number']).drop(columns=['label_binary'], errors='ignore').fillna(0).values
    iso = models.get('iso')
    ae = models.get('ae')
    if iso is None and ae is None:
        return "No models trained. Run training script first.", None
    scores = []
    if iso is not None:
        scores.append(anomaly_score_isoforest(iso, X))
    if ae is not None:
        scores.append(anomaly_score_autoencoder(ae, X))
    # average scores
    arr = np.vstack(scores) if len(scores) > 0 else np.zeros((1, X.shape[0]))
    avg = np.mean(arr, axis=0)

    ts = df_proc.reset_index()[['timestamp']].copy()
    ts['score'] = avg
    ts_plot = ts.set_index('timestamp')['score'].resample('T').mean().fillna(0)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ts_plot.index, ts_plot.values)
    ax.set_title("Aggregated Anomaly Score Over Time")
    ax.set_ylabel("Anomaly Score")
    ax.set_xlabel("Time")
    return None, fig

# Gradio UI components
with gr.Blocks() as demo:
    gr.Markdown("# Advanced Anomaly Detection — Demo (Gradio)")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Inspect sample data")
            sample_data_btn = gr.Button("Show sample preprocessed rows")
            sample_output = gr.Dataframe(headers=None, interactive=False)
        with gr.Column():
            gr.Markdown("## Real-time detection")
            # create default sample dict from raw pipeline columns
            df = load_raw(cfg)
            df_proc, _, _ = preprocess(df, cfg, save_encoders=False)
            sample_columns = df_proc.select_dtypes(include=['number']).columns.tolist()
            default_sample = {c: float(df_proc.iloc[0][c]) if c in df_proc.columns else 0.0 for c in sample_columns}
            input_json = gr.JSON(value=default_sample, label="Input sample (feature->value)")
            detect_btn = gr.Button("Run detection")
            detect_out = gr.JSON()

    sample_data_btn.click(fn=inspect_data, inputs=[], outputs=[sample_output])
    detect_btn.click(fn=run_detection, inputs=[input_json], outputs=[detect_out])

    gr.Markdown("## Time-series aggregated anomaly score")
    err_msg = gr.Textbox(interactive=False)
    plot_output = gr.Plot()
    plot_btn = gr.Button("Show aggregated anomaly score (plot)")
    plot_btn.click(fn=show_scores_time_series, inputs=[gr.Slider(minimum=10, maximum=1000, step=10, value=120)], outputs=[err_msg, plot_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
