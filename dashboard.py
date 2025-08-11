# Import library
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, sqlite3
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from tensorflow.keras.models import load_model

DB_PATH = "forecast.sqlite"
TZ = "Asia/Jakarta"

# URL ThingSpeak API
url = "https://api.thingspeak.com/channels/2990169/feeds.json"
params = {
    "api_key": "LDXFP3LRNTBZCFMU",
    "results": 100  # Ambil 120 data terakhir
}

look_back = 20  # Jumlah data historis untuk prediksi
features = ['PM2.5', 'PM10', 'CO', 'CO2']
n_features = len(features)

# Load model LSTM jika tersedia
try:
    lstm_model = load_model("model-1.h5", compile=False)
    scaler = joblib.load("scaler.save")
    scaler_X = joblib.load("scaler_X.save")
    scaler_y = joblib.load("scaler_y.save")
except:
    lstm_model = None

# Konfigurasi metrik polutan
metrics = {
    "temperature": {"display": "Suhu (°C)", "field": 1, "color": "brown"},
    "humidity": {"display": "Kelembapan (%)", "field": 2, "color": "purple"},
    "pm25": {"display": "PM2.5 (μg/m³)", "field": 3, "color": "blue"},
    "pm10": {"display": "PM10 (μg/m³)", "field": 4, "color": "green"},
    "co":   {"display": "CO (ppm)", "field": 5, "color": "red"},
    "co2":  {"display": "CO₂ (ppm)", "field": 6, "color": "orange"},
}

forecast_metrics = {
    "pm25": {"display": "PM2.5 (μg/m³)", "field": 3, "color": "blue"},
    "pm10": {"display": "PM10 (μg/m³)", "field": 4, "color": "green"},
    "co":   {"display": "CO (ppm)", "field": 5, "color": "red"},
    "co2":  {"display": "CO₂ (ppm)", "field": 6, "color": "orange"},
}

# Inisialisasi aplikasi Dash
app = Dash(__name__)
app.title = "Dashboard Polutan + Prediksi"

# Layout halaman web Dash
app.layout = html.Div([
    # Tambahkan link Google Fonts
    html.Link(
        rel="stylesheet",
        href="https://fonts.googleapis.com/css?family=Poppins:400,600&display=swap"
    ),
    html.H1("Dashboard Polutan Aktual + Prediksi", style={"textAlign": "center"}),

    # Kotak metrik aktual
    html.Div(id="latest-metrics", children=[
        html.Div(id=f"metric-{key}", style={
            "border": "1px solid #ccc", "padding": "20px", "textAlign": "center",
            "borderRadius": "10px", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
            "backgroundColor": "#f9f9f9"
        }) for key in metrics.keys()
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "margin": "20px"}),

    # Grafik aktual
    html.Div([
        *[dcc.Graph(id=f"graph-{key}", style={"height": "400px"}) for key in metrics.keys()]
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}),

    # Judul prediksi
    html.H2("Prediksi 1 Jam Ke Depan", style={"textAlign": "center", "marginTop": "30px"}),

    # Kotak metrik prediksi
    html.Div(id="forecast-metrics", children=[
        html.Div(id=f"forecast-metric-{key}", style={
            "border": "1px solid #ccc", "padding": "20px", "textAlign": "center",
            "borderRadius": "10px", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
            "backgroundColor": "#fffaf0"
        }) for key in forecast_metrics.keys()
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "20px", "margin": "20px"}),

    # Grafik prediksi
    html.Div([
        *[dcc.Graph(id=f"graph-{key}-forecast", style={"height": "400px"}) for key in forecast_metrics.keys()]
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}),

    # Menampilkan judul untuk metrik evaluasi
    html.H2("Metrik Evaluasi", style={"textAlign": "center", "marginTop": "30px"}),

    # Membuat kotak untuk membandingkan metrik prediksi dengan nilai aktual
    html.Div(id="forecast-metrics-compare", children=[
        html.Div(id=f"forecast-metric-{key}-compare", style={
            "border": "1px solid #ccc", "padding": "20px", "textAlign": "center",
            "borderRadius": "10px", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
            "backgroundColor": "#fffaf0"
        }) for key in forecast_metrics.keys()
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "20px", "margin": "20px"}),

    # Interval update setiap 60 detik
    dcc.Interval(id="interval-component", interval=60 * 1000, n_intervals=0)
], style={"fontFamily": "Poppins, sans-serif"})

# Fungsi ambil data dari ThingSpeak
def fetch_data():
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        now_wib = datetime.utcnow() + timedelta(hours=7)
        one_hour_ago = now_wib - timedelta(hours=1)

        timestamps = []
        values = {key: [] for key in metrics.keys()}

        for feed in data["feeds"]:
            utc_dt = datetime.strptime(feed["created_at"], "%Y-%m-%dT%H:%M:%SZ")
            wib_dt = utc_dt + timedelta(hours=7)
            if wib_dt >= one_hour_ago:
                timestamps.append(wib_dt)
                for key, meta in metrics.items():
                    field_val = feed.get(f"field{meta['field']}")
                    values[key].append(float(field_val) if field_val else np.nan)

        dfs = {}
        for key in metrics.keys():
            dfs[key] = pd.Series(values[key], index=pd.to_datetime(timestamps)).dropna()
        return dfs
    else:
        return {}

def db():
    return sqlite3.connect(DB_PATH, timeout=10)

def init_db():
    with db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS forecast_data (
        zid INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_target TEXT NOT NULL,   -- 'YYYY-MM-DD HH:00:00' (jam bulat)
        feature TEXT NOT NULL,   -- 'PM2.5'/'PM10'/'CO'/'CO2'
        y_pred REAL NOT NULL,   -- prediksi untuk ts_target (disimpan 1x)
        y_true REAL,
        mape REAL,
        compared  INTEGER DEFAULT 0
        );
        """)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_forecast_data ON forecast_data(ts_target, feature);")

def insert(ts_target_str, feature, ypred):
    with db() as conn:
        conn.execute("""
        INSERT OR IGNORE INTO forecast_data (ts_target, feature, y_pred)
        VALUES (?, ?, ?);
        """, (ts_target_str, feature, float(ypred)))

def compare(actual_df):
    with db() as conn:
        rows = conn.execute("""
        SELECT id, ts_target, feature, y_pred
        FROM forecast_data
        WHERE compared = 0
        ORDER BY ts_target;
        """).fetchall()
    if not rows:
        return

    actual = actual_df.copy()
    actual.index = actual.index.floor('min')
    actual = actual[~actual.index.duplicated(keep='last')]

    updates = []
    for row_id, ts_target, feat, y_pred in rows:
        t = pd.to_datetime(ts_target).floor('min')
        if t not in actual.index.floor('min'):
            continue
        y_true = float(actual.loc[t, feat])
        error = modified_mape(y_true, y_pred)
        updates.append((y_true, error, 1, row_id))

    if updates:
        with db() as conn:
            conn.executemany("""
            UPDATE forecast_data
            SET y_true = ?, mape = ?, compared = ?
            WHERE id = ?;
            """, updates)
        
def get_metric_data(feature):
    with db() as conn:
        row = conn.execute("""
        SELECT mape
        FROM forecast_data
        WHERE feature = ? AND compared = 1 AND mape IS NOT NULL
        ORDER BY ts_target DESC
        """, (feature,)).fetchone()
    return None if not row or row[0] is None else float(row[0])

init_db()

# Fungsi buat grafik aktual
def generate_figure(series, name, color):
    if series.empty:
        return go.Figure(layout={"title": f"Gagal ambil data: {name}"})

    trace_actual = go.Scatter(x=series.index, y=series.values, mode="lines+markers",
                              name=f"{name} Aktual", line=dict(color=color))

    layout = go.Layout(title=name, xaxis={"title": "Waktu", "tickformat": "%H:%M"},
                       yaxis={"title": name}, hovermode="closest",
                       legend=dict(orientation="h"))

    return go.Figure(data=[trace_actual], layout=layout)

# Fungsi prediksi
def generate_forecast(dfs, key):
    try:
        # Ambil data untuk key yang diberikan
        df = pd.DataFrame(dfs).rename(columns={
            'pm25': 'PM2.5',
            'pm10': 'PM10',
            'co': 'CO',
            'co2': 'CO2'}).dropna()
        # Resample data menjadi interval 3 menit
        df.index = df.index.floor('min')
        df_resampled = df[df.index.minute % 3 == 0].copy()

        df_resampled['hour'] = df_resampled.index.hour
        df_resampled['minute'] = df_resampled.index.minute
        df_resampled['dayofweek'] = df_resampled.index.dayofweek

        # Cek apakah data cukup untuk prediksi
        if len(df_resampled) < look_back:
            return go.Figure(layout={"title": f"Tidak cukup data untuk prediksi: {key}"})

        recent_data = df_resampled.tail(look_back)

        # Jika model LSTM tersedia, gunakan untuk prediksi
        if lstm_model:
            seq = recent_data.copy()
            scaled = scaler_X.transform(seq)
            input_seq = scaled.reshape(1, look_back, 7)
            pred_scaled = lstm_model.predict(input_seq, verbose=0)
            pred_scaled = pred_scaled.reshape(-1, n_features)
            pred = scaler_y.inverse_transform(pred_scaled)
            # print(pred)
            # Buat array prediksi
            pred_arr = np.array(pred)
            # print(pred_arr)
            # Buat daftar waktu prediksi
            times = [df_resampled.index[-1] + timedelta(minutes=3 * i) for i in range(look_back)]
            # Map key ke fitur model
            key_map = {
                'pm25': 'PM2.5',
                'pm10': 'PM10',
                'co': 'CO',
                'co2': 'CO2'
            }
            # Ambil model key dari key_map
            model_key = key_map.get(key)

            # Temukan indeks fitur dalam daftar features
            idx = features.index(model_key)

            # Buat grafik prediksi
            return go.Figure(data=[go.Scatter(
                x=times,
                y=pred_arr[:, idx],
                mode='lines+markers',
                name=f"Prediksi {metrics[key]['display']}",
                line=dict(color='orange')
            )], layout=go.Layout(
                title=f"{metrics[key]['display']} Prediksi 1 Jam",
                xaxis={"title": "Waktu", "tickformat": "%H:%M"},
                yaxis={"title": metrics[key]['display']},
                hovermode="closest"
            ))
        else:
        # Fallback jika model tidak ada
            # Gunakan metode sederhana untuk prediksi
            s = df_resampled[key]
            # Hitung perbedaan antar nilai
            diffs = s.diff().dropna()
            # Jika tidak ada cukup data, gunakan data acak
            if len(diffs) < 5:
                diffs = pd.Series(np.random.normal(0, 0.01, 10))

            # Ambil pola dari 10 perbedaan terakhir
            pattern = diffs[-10:].values.flatten()
            # Sesuaikan pola berdasarkan jenis polutan
            pattern *= {"pm25": 1.0, "pm10": 1.2, "co": 0.5, "co2": 0.1}.get(key, 1.0)

            # Buat prediksi berdasarkan pola
            last_val = s.iloc[-1]
            pred_vals = [last_val]
            for i in range(20):
                delta = pattern[i % len(pattern)]
                noise = np.random.normal(0, abs(delta) * 0.2)
                next_val = max(pred_vals[-1] + delta + noise, 0)
                pred_vals.append(next_val)

            future_times = [s.index[-1] + timedelta(minutes=10 * (i + 1)) for i in range(6)]

            return go.Figure(data=[go.Scatter(
                x=future_times,
                y=pred_vals[1:],
                mode="lines+markers",
                name=f"Prediksi {metrics[key]['display']}",
                line=dict(color="orange")
            )], layout=go.Layout(
                title=f"Prediksi {metrics[key]['display']} 1 Jam",
                xaxis={"title": "Waktu", "tickformat": "%H:%M"},
                yaxis={"title": metrics[key]['display']}
            ))

    except Exception as e:
        return go.Figure(layout={"title": f"Error prediksi: {str(e)}"})

# CALLBACK untuk grafik aktual
def create_actual_callback(metric_key):
    @app.callback(
        Output(f"graph-{metric_key}", "figure"),
        Input("interval-component", "n_intervals"),
        prevent_initial_call="initial_duplicate"
    )
    def update_graph(n):
        dfs = fetch_data()
        if metric_key in dfs:
            return generate_figure(dfs[metric_key], metrics[metric_key]["display"], metrics[metric_key]["color"])
        else:
            return go.Figure(layout={"title": f"Gagal ambil data: {metrics[metric_key]['display']}"})

# CALLBACK untuk grafik prediksi
def create_forecast_callback(metric_key):
    @app.callback(
        Output(f"graph-{metric_key}-forecast", "figure"),
        Input("interval-component", "n_intervals"),
        prevent_initial_call="initial_duplicate"
    )
    def update_forecast(n):
        dfs = fetch_data()
        for key in list(dfs.keys()):
            if key not in forecast_metrics.keys():
                dfs.pop(key)
        # Pastikan hanya mengambil metrik yang relevan untuk prediksi
        if metric_key in dfs and metric_key:
            return generate_forecast(dfs, metric_key)
        else:
            return go.Figure(layout={"title": f"Gagal ambil data: {metrics[metric_key]['display']}"})

# CALLBACK untuk metrik aktual
for key in metrics.keys():
    @app.callback(
        Output(f"metric-{key}", "children"),
        Input("interval-component", "n_intervals"),
        prevent_initial_call="initial_duplicate"
    )
    def update_metric(n, key=key):
        dfs = fetch_data()
        if key in dfs and not dfs[key].empty:
            latest_val = round(dfs[key].iloc[-1], 2)
            return html.Div([
                html.H4(metrics[key]["display"]),
                html.H2(f"{latest_val}", style={"color": metrics[key]["color"], "fontSize": "36px"})
            ])
        else:
            return html.Div([
                html.H4(metrics[key]["display"]),
                html.H2("N/A", style={"color": "gray"})
            ])

def modified_mape(y_true, y_pred, epsilon=1e-2):
    y_true = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    return max(0, 100 - mape)

# CALLBACK untuk metrik prediksi
def forecasting(key, metric_type):
    output_id = f"forecast-metric-{key}" if metric_type == "pred" else f"forecast-metric-{key}-compare"

    @app.callback(
        Output(output_id, "children"),
        Input("interval-component", "n_intervals"),
        prevent_initial_call="initial_duplicate"
    )
    def update_forecast_metric(n):
        dfs = fetch_data()
        dfs = {k: v for k, v in dfs.items() if k in forecast_metrics}

        if key in dfs and not dfs[key].empty:
            df = pd.DataFrame(dfs).rename(columns={
                'pm25': 'PM2.5',
                'pm10': 'PM10',
                'co': 'CO',
                'co2': 'CO2'
            }).dropna()

            df.index = df.index.floor('min')
            df_resampled = df[df.index.minute % 3 == 0].copy()
            df_resampled['hour'] = df_resampled.index.hour
            df_resampled['minute'] = df_resampled.index.minute
            df_resampled['dayofweek'] = df_resampled.index.dayofweek

            if len(df_resampled) >= look_back and lstm_model:
                recent = df_resampled.tail(look_back).values
                scaled = scaler_X.transform(recent.copy())
                input_seq = scaled.reshape((1, look_back, 7))
                pred_scaled = lstm_model.predict(input_seq, verbose=0)
                pred_scaled = pred_scaled.reshape(-1, n_features)
                pred = scaler_y.inverse_transform(pred_scaled)

                last_idx = df_resampled.index[-1]
                ts_target = last_idx + pd.Timedelta(minutes=60)
                ts_target_str = ts_target.strftime("%Y-%m-%d %H:%M:%S")

                for i, feat in enumerate(features):
                    insert(ts_target_str, feat, pred[-1, i])

                compare(df_resampled)

                if metric_type == 'pred':
                    key_map = {
                        'pm25': 'PM2.5',
                        'pm10': 'PM10',
                        'co': 'CO',
                        'co2': 'CO2'
                    }
                    model_key = key_map.get(key)
                    idx = features.index(model_key)
                    final_val = pred[-1, idx]
                    return html.Div([
                        html.H4(f"Prediksi {metrics[key]['display']}"),
                        html.H2(f"{final_val:.2f}", style={"color": "orange", "fontSize": "36px"})
                    ])
                else:  # metric_type == 'mae'
                    key_map = {
                        'pm25': 'PM2.5',
                        'pm10': 'PM10',
                        'co': 'CO',
                        'co2': 'CO2'
                    }
                    feat = key_map.get(key)
                    mape = get_metric_data(feat)
                    if mape is None:
                        return html.Div([
                            html.H4(f"Akurasi {metrics[key]['display']}"),
                            html.H2("Menunggu data real pada menit target…",
                                    style={"color": "gray", "fontSize": "20px"})
                        ])
                    return html.Div([
                        html.H4(f"Akurasi {metrics[key]['display']}"),
                        html.H2(f"{mape:.2f}%", style={"color": "orange", "fontSize": "36px"})
                    ])

        return html.Div([
            html.H4(f"Prediksi {metrics[key]['display']}"),
            html.H2("N/A", style={"color": "gray"})
        ])

# Register callbacks
for k in forecast_metrics.keys():
    forecasting(k, 'pred')
    forecasting(k, 'mae')

# Registrasi semua callback grafik
for key in metrics.keys():
    create_actual_callback(key)
for key in forecast_metrics.keys():
    create_forecast_callback(key)

# Jalankan aplikasi
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8001)
