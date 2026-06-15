import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import xgboost as xgb


st.set_page_config(
    page_title="Indian Roads Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "prediction_data" not in st.session_state:
    st.session_state["prediction_data"] = None


st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252b40);
        border: 1px solid #3a3f5c;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h3 {
        color: #8b92b4; font-size: 13px; margin: 0 0 6px 0;
        letter-spacing: 0.8px; text-transform: uppercase;
    }
    .metric-card h1 { color: #e8eaf6; font-size: 32px; margin: 0; font-weight: 700; }
    .metric-card p  { color: #7c83a8; font-size: 12px; margin: 4px 0 0 0; }

    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 24px; font-weight: 700; margin-bottom: 8px;
    }
    .pred-high   { background:#ff4b4b22; border:2px solid #ff4b4b; border-radius:10px; padding:16px; text-align:center; }
    .pred-medium { background:#ffa50022; border:2px solid #ffa500; border-radius:10px; padding:16px; text-align:center; }
    .pred-low    { background:#00cc6622; border:2px solid #00cc66; border-radius:10px; padding:16px; text-align:center; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 46px; border-radius: 8px 8px 0 0;
        background: #1e2130; color: #8b92b4;
        padding: 0 20px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #667eea !important; color: #fff !important;
    }
    .insight-box {
        background: #1a1f2e; border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0;
        color: #c5c9e0; font-size: 14px;
    }
            
    .header {
    text-align: center;
    font-size: 2.7rem;
    font-weight:600;
    color: white;
    background: linear-gradient(90deg, #1f37b4, #17becf);
    border-radius: 12px; 
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    padding: 30px;
    }
    
    .block-container {
    padding-top: 2.8rem;
}
</style>
""", unsafe_allow_html=True)


CITY_COORDS = {
    "Bangalore":  (12.9716, 77.5946),
    "Chandigarh": (30.7333, 76.7794),
    "Chennai":    (13.0827, 80.2707),
    "Delhi":      (28.6139, 77.2090),
    "Hyderabad":  (17.3850, 78.4867),
    "Kolkata":    (22.5726, 88.3639),
    "Mumbai":     (19.0760, 72.8777),
    "Pune":       (18.5204, 73.8567),
}

TRAFFIC_COLOR = {"high": "#ff4b4b", "medium": "#ffa500", "low": "#00cc66"}
TRAFFIC_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}


@st.cache_data
def load_data():
    df = pd.read_csv("indian_roads_dataset.csv")
    df["date"]          = pd.to_datetime(df["date"], errors="coerce")
    df["month"]         = df["date"].dt.month
    df["month_name"]    = df["date"].dt.strftime("%b")
    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    """
    End-to-end pipeline:
      • Auto-detect numeric / categorical columns from X
      • Impute → Scale (numeric) / Impute → OneHotEncode (categorical)
      • XGBoost with RandomizedSearchCV (20 iters, 5-fold CV)
      • GradientBoosting as secondary model
    Returns everything the UI needs.
    """
    target_col = "traffic_density"
    
    drop_cols = [
        target_col,
        "accident_id", "date", "festival", "festival_name", "month_name","state","time", "is_festival"
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y_raw = df[target_col].astype(str)

    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw)
    
   # for i, label in enumerate(le_target.classes_):
    #    st.write(f"{label} -> {i}")

    numeric_features     = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer,     numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb_base = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        classes = 'balanced',
        reg_alpha=0.01,
        random_state=42,
    )

    xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   xgb_base),
    ])

    param_grid = {
    'classifier__n_estimators': [100, 300, 500, 800],
    'classifier__max_depth': [2,3, 5, 7, 9],
    'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'classifier__min_child_weight': [1, 3, 5],
    'classifier__gamma': [0, 0.1, 0.3],
    'classifier__reg_alpha': [0, 0.01, 0.1, 1],
    'classifier__reg_lambda': [1, 2, 5]
    }

    with st.spinner("🔍 Running RandomizedSearchCV for XGBoost (20 iters × 5-fold)…"):
        random_search = RandomizedSearchCV(
            xgb_pipeline,
            param_distributions=param_grid,
            n_iter=20,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
        )

        random_search.fit(X_train, y_train)

    best_xgb_pipeline = random_search.best_estimator_
    best_params = random_search.best_params_

    xgb_train_acc = best_xgb_pipeline.score(X_train, y_train)
    xgb_test_acc  = best_xgb_pipeline.score(X_test,  y_test)
    y_pred_xgb    = best_xgb_pipeline.predict(X_test)
    xgb_cv        = random_search.best_score_          
    xgb_cm        = confusion_matrix(y_test, y_pred_xgb)
    xgb_report    = classification_report(
        y_test, y_pred_xgb,
        target_names=le_target.classes_,
        output_dict=True,
    )

    if xgb_train_acc > xgb_test_acc + 0.05:
        xgb_diagnosis = "⚠️ Overfitting — try reducing max_depth or increasing reg_alpha."
    elif xgb_train_acc < 0.7 and xgb_test_acc < 0.7:
        xgb_diagnosis = "⚠️ Underfitting — add more features or a more complex model."
    else:
        xgb_diagnosis = "✅ Model performance is stable."


    fitted_preprocessor = best_xgb_pipeline.named_steps["preprocessor"]
    ohe_cats = (fitted_preprocessor
                .named_transformers_["cat"]
                .named_steps["onehot"]
                .get_feature_names_out(categorical_features)
                .tolist())
    all_feature_names = numeric_features + ohe_cats

    importances = best_xgb_pipeline.named_steps["classifier"].feature_importances_
    feat_imp = (
        pd.DataFrame({"Feature": all_feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    trained_models = {
        "XGBoost":    best_xgb_pipeline
    }
    model_results = {
        "XGBoost": {
            "accuracy":    xgb_test_acc,
            "train_acc":   xgb_train_acc,
            "cv_score":    xgb_cv,
            "cm":          xgb_cm,
            "report":      xgb_report,
            "diagnosis":   xgb_diagnosis,
            "best_params": best_params,
        },
    }

    return (
        trained_models,
        model_results,
        le_target,
        feat_imp,
        numeric_features,
        categorical_features,
        X_test,
        y_test,
    )


df = load_data()

(
    trained_models,
    model_results,
    le_target,
    feat_imp,
    numeric_features,
    categorical_features,
    X_test_raw,
    y_test,
) = train_models(df)


with st.sidebar:
    st.markdown("## 🚦 Traffic Intelligence")
    st.markdown(" ")
    st.markdown(f"📍 **Cities:** {df['city'].nunique()}")
    st.markdown(f"🗺️ **States:** {df['state'].nunique()}")
    st.markdown(f"📅 **Features:** {df.shape[1]}")
    st.markdown(" ")

    st.markdown("### 🔧 Global Filters")
    sel_cities  = st.multiselect("Select Cities",  sorted(df["city"].unique()),    default=sorted(df["city"].unique()))
    sel_road    = st.multiselect("Road Type",       df["road_type"].unique(),       default=list(df["road_type"].unique()))
    sel_weather = st.multiselect("Weather",         df["weather"].unique(),         default=list(df["weather"].unique()))

    st.markdown(" ")
    st.markdown("### 🤖 Active ML Model")
    active_model = st.selectbox("Model", list(trained_models.keys()))

# filtered data
dff = df[
    df["city"].isin(sel_cities) &
    df["road_type"].isin(sel_road) &
    df["weather"].isin(sel_weather)
].copy()


st.markdown(
    '<div class="header">🚦 Traffic Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)


tabs = st.tabs([
    "📊 Overview",
    "🗺️ Geospatial Analysis",
    "📈 EDA",
    "🤖 ML Models",
    "🔮 Predict Traffic",
])

with tabs[0]:
    st.markdown('<p class="section-header">📊 Dataset Overview & KPIs</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    high_pct = round(len(dff[dff["traffic_density"] == "high"]) / max(len(dff), 1) * 100, 1)
    avg_risk = round(dff["risk_score"].mean(), 3)
    avg_cas  = round(dff["casualties"].mean(), 2)
    peak_pct = round(dff["is_peak_hour"].mean() * 100, 1)

    for col, val, label, note in zip(
        [c1, c2, c3, c4, c5],
        [f"{len(dff):,}", f"{high_pct}%", f"{avg_risk}", f"{avg_cas}", f"{peak_pct}%"],
        ["Total Records", "High Traffic", "Avg Risk Score", "Avg Casualties", "Peak Hour %"],
        ["filtered", "density", "0–1 scale", "per incident", "of records"],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{label}</h3><h1>{val}</h1><p>{note}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        td_counts = dff["traffic_density"].value_counts().reset_index()
        td_counts.columns = ["Traffic Density", "Count"]
        fig = px.pie(td_counts, names="Traffic Density", values="Count",
                     color="Traffic Density", color_discrete_map=TRAFFIC_COLOR,
                     hole=0.5, title="Traffic Density Distribution")
        fig.update_layout(template="plotly_dark", height=350,
                          plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        city_td = dff.groupby(["city", "traffic_density"]).size().reset_index(name="Count")
        fig = px.bar(city_td, x="city", y="Count", color="traffic_density",
                     color_discrete_map=TRAFFIC_COLOR, barmode="stack",
                     title="Traffic Density by City")
        fig.update_layout(template="plotly_dark", height=350,
                          plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                          xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        hourly = dff.groupby(["hour", "traffic_density"]).size().reset_index(name="Count")
        fig = px.line(hourly, x="hour", y="Count", color="traffic_density",
                      color_discrete_map=TRAFFIC_COLOR,
                      title="Hourly Traffic Pattern", markers=True)
        fig.update_layout(template="plotly_dark", height=330,
                          plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        DOW = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_td = dff.groupby(["day_of_week", "traffic_density"]).size().reset_index(name="Count")
        dow_td["day_of_week"] = pd.Categorical(dow_td["day_of_week"], categories=DOW, ordered=True)
        dow_td = dow_td.sort_values("day_of_week")
        fig = px.bar(dow_td, x="day_of_week", y="Count", color="traffic_density",
                     color_discrete_map=TRAFFIC_COLOR, barmode="group",
                     title="Day-of-Week Traffic")
        fig.update_layout(template="plotly_dark", height=330,
                          plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                          xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🗃️ View Raw Data"):
        st.dataframe(dff.head(500), use_container_width=True)
        st.caption(f"Showing 500 of {len(dff):,} filtered rows.")



with tabs[1]:
    st.markdown('<p class="section-header">🗺️ Geospatial Traffic Intelligence</p>', unsafe_allow_html=True)

    map_type = st.radio(
        "Map Visualization",
        ["📍 Accident Scatter Map", "🔥 Heatmap by Traffic Density", "🏙️ City Summary Bubbles"],
        horizontal=True,
    )
    col_m1, col_m2 = st.columns([3, 1])

    with col_m2:
        st.markdown("### 🎛️ Map Controls")
        map_city = st.multiselect("Cities", sorted(dff["city"].unique()),
                                  default=sorted(dff["city"].unique()), key="map_city")
        map_td   = st.multiselect("Traffic Density", ["high", "medium", "low"],
                                  default=["high", "medium", "low"])
        sample_n = st.slider("Max Points (Scatter)", 500, 5000, 2000, 500)

    with col_m1:
        map_df = dff[dff["city"].isin(map_city) & dff["traffic_density"].isin(map_td)].copy()

        if map_type == "📍 Accident Scatter Map":
            plot_df = map_df.sample(min(sample_n, len(map_df)), random_state=42)
            fig = px.scatter_mapbox(
                plot_df, lat="latitude", lon="longitude",
                color="traffic_density", color_discrete_map=TRAFFIC_COLOR,
                size="risk_score", size_max=14,
                hover_data={"city": True, "road_type": True, "weather": True,
                            "accident_severity": True, "risk_score": ":.2f",
                            "latitude": False, "longitude": False},
                zoom=4.5, center={"lat": 20.5, "lon": 79.0},
                mapbox_style="carto-darkmatter",
                title=f"Accident Scatter Map ({len(plot_df):,} points)",
            )
            fig.update_layout(template="plotly_dark", height=560,
                              paper_bgcolor="#0e1117", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        elif map_type == "🔥 Heatmap by Traffic Density":
            fig = go.Figure()
            for td, color in TRAFFIC_COLOR.items():
                sub = map_df[map_df["traffic_density"] == td]
                if len(sub) < 2:
                    continue
                fig.add_trace(go.Densitymapbox(
                    lat=sub["latitude"], lon=sub["longitude"],
                    z=sub["risk_score"], radius=25,
                    name=td.capitalize(),
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, color]],
                    showscale=False, opacity=0.75,
                ))
            fig.update_layout(
                mapbox_style="carto-darkmatter",
                mapbox_center={"lat": 20.5, "lon": 79.0},
                mapbox_zoom=4, height=560,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="#0e1117",
                title="Density Heatmap by Traffic Level",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            city_agg = (
                map_df.groupby("city")
                .agg(total=("accident_id","count"),
                     high=("traffic_density", lambda x: (x=="high").sum()),
                     avg_risk=("risk_score","mean"),
                     avg_cas=("casualties","mean"))
                .reset_index()
            )
            city_agg["lat"]      = city_agg["city"].map(lambda c: CITY_COORDS.get(c, (20,79))[0])
            city_agg["lon"]      = city_agg["city"].map(lambda c: CITY_COORDS.get(c, (20,79))[1])
            city_agg["high_pct"] = (city_agg["high"] / city_agg["total"] * 100).round(1)

            fig = px.scatter_mapbox(
                city_agg, lat="lat", lon="lon",
                size="total", color="avg_risk",
                color_continuous_scale="RdYlGn_r",
                hover_name="city",
                hover_data={"total": True, "high_pct": True,
                            "avg_risk": ":.3f", "avg_cas": ":.2f",
                            "lat": False, "lon": False},
                size_max=50, zoom=4, center={"lat": 20.5, "lon": 79.0},
                mapbox_style="carto-darkmatter",
                title="City-Level Traffic Risk Bubbles",
            )
            fig.update_layout(height=560, paper_bgcolor="#0e1117",
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Geospatial Statistics by City")
    city_stats = (
        dff.groupby("city")
        .agg(Incidents=("accident_id","count"),
             High_Traffic=("traffic_density", lambda x: (x=="high").sum()),
             Avg_Risk=("risk_score","mean"),
             Avg_Casualties=("casualties","mean"),
             Peak_Hour_Pct=("is_peak_hour","mean"))
        .reset_index()
    )
    city_stats["High_Traffic_%"] = (city_stats["High_Traffic"] / city_stats["Incidents"] * 100).round(1)
    city_stats["Avg_Risk"]       = city_stats["Avg_Risk"].round(3)
    city_stats["Avg_Casualties"] = city_stats["Avg_Casualties"].round(2)
    city_stats["Peak_Hour_Pct"]  = (city_stats["Peak_Hour_Pct"] * 100).round(1)
    city_stats = city_stats.drop("High_Traffic", axis=1)
    st.dataframe(city_stats.sort_values("Avg_Risk", ascending=False),
                 use_container_width=True, hide_index=True)

    # Folium map
    st.markdown("### 🗺️ Interactive Folium Map — City Risk Summary")
    m = folium.Map(location=[20.5, 79.0], zoom_start=5, tiles="cartodb positron")

    for _, row in city_stats.iterrows():
        city = row["city"]
        lat, lon = CITY_COORDS.get(city, (20, 79))
        risk  = row["Avg_Risk"]
        color = "#ff4b4b" if risk > 0.55 else ("#ffa500" if risk > 0.4 else "#00cc66")
        popup_html = f"""
        <div style='font-family:Arial; min-width:180px;'>
            <b style='font-size:15px;'>{city}</b><br>
            📦 Incidents: <b>{int(row['Incidents'])}</b><br>
            ⚠️ Avg Risk: <b>{risk:.3f}</b><br>
            🔴 High Traffic: <b>{row['High_Traffic_%']}%</b><br>
            🚑 Avg Casualties: <b>{row['Avg_Casualties']}</b><br>
            ⏰ Peak Hour %: <b>{row['Peak_Hour_Pct']}%</b>
        </div>"""
        folium.CircleMarker(
            location=[lat, lon],
            radius=max(8, row["Incidents"] / 350),
            color=color, fill=True, fill_color=color,
            fill_opacity=0.8, weight=2,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{city} | Risk: {risk:.3f}",
        ).add_to(m)

    legend = """
    <div style='position:fixed; bottom:30px; left:30px; z-index:1000;
                background:#1e2130; padding:12px 18px; border-radius:10px;
                border:1px solid #444; color:#fff; font-size:12px;'>
        <b>Risk Level</b><br>🔴 High (&gt;0.55)<br>🟡 Medium (0.4–0.55)<br>🟢 Low (&lt;0.4)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    st_folium(m, width=None, height=480)



with tabs[2]:
    st.markdown('<p class="section-header">📈 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    eda_choice = st.selectbox("EDA Section", [
        "Weather & Visibility Analysis",
        "Road Type & Cause Analysis",
        "Temporal Patterns",
        "Risk Score Distributions",
        "Correlation Heatmap",
        "Severity Analysis",
    ])

    if eda_choice == "Weather & Visibility Analysis":
        col1, col2 = st.columns(2)
        with col1:
            wt = dff.groupby(["weather", "traffic_density"]).size().reset_index(name="Count")
            fig = px.bar(wt, x="weather", y="Count", color="traffic_density",
                         color_discrete_map=TRAFFIC_COLOR, barmode="group",
                         title="Weather vs Traffic Density")
            fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            vis = dff.groupby(["visibility", "traffic_density"]).size().reset_index(name="Count")
            fig = px.bar(vis, x="visibility", y="Count", color="traffic_density",
                         color_discrete_map=TRAFFIC_COLOR, barmode="stack",
                         title="Visibility vs Traffic Density")
            fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="insight-box">💡 <b>Insight:</b> Fog conditions show significantly higher high-traffic incidents, while clear weather correlates with more low-traffic events.</div>', unsafe_allow_html=True)

        fig = px.sunburst(dff, path=["weather","visibility","traffic_density"],
                          color="traffic_density", color_discrete_map=TRAFFIC_COLOR,
                          title="Weather → Visibility → Traffic Density Hierarchy")
        fig.update_layout(template="plotly_dark", height=450, paper_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

    elif eda_choice == "Road Type & Cause Analysis":
        col1, col2 = st.columns(2)
        with col1:
            rt = dff.groupby(["road_type","traffic_density"]).size().reset_index(name="Count")
            fig = px.bar(rt, x="road_type", y="Count", color="traffic_density",
                         color_discrete_map=TRAFFIC_COLOR, barmode="group",
                         title="Road Type vs Traffic Density")
            fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            cause = dff.groupby(["cause","traffic_density"]).size().reset_index(name="Count")
            fig = px.bar(cause, x="cause", y="Count", color="traffic_density",
                         color_discrete_map=TRAFFIC_COLOR, barmode="stack",
                         title="Accident Cause vs Traffic Density")
            fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117",
                              paper_bgcolor="#0e1117", xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

        treemap_df = (dff.groupby(["road_type","cause","traffic_density"])
                      .size().reset_index(name="Count"))
        fig = px.treemap(treemap_df, path=["road_type","cause","traffic_density"],
                         values="Count", color="traffic_density",
                         color_discrete_map=TRAFFIC_COLOR,
                         title="Road Type → Cause → Traffic Density Treemap")
        fig.update_layout(paper_bgcolor="#0e1117", height=420)
        st.plotly_chart(fig, use_container_width=True)

    elif eda_choice == "Temporal Patterns":
        col1, col2 = st.columns(2)
        with col1:
            mnth = dff.groupby(["month","traffic_density"]).size().reset_index(name="Count")
            fig = px.line(mnth, x="month", y="Count", color="traffic_density",
                          color_discrete_map=TRAFFIC_COLOR, markers=True,
                          title="Monthly Traffic Density Trend")
            fig.update_layout(
                template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                xaxis=dict(tickvals=list(range(1,13)),
                           ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                                     "Jul","Aug","Sep","Oct","Nov","Dec"]),
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            hr_city = dff.groupby(["hour","city"])["risk_score"].mean().reset_index()
            fig = px.line(hr_city, x="hour", y="risk_score", color="city",
                          title="Average Risk Score by Hour & City")
            fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)

        heatmap_data = dff.groupby(["day_of_week","hour"]).size().unstack(fill_value=0)
        DOW = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        heatmap_data = heatmap_data.reindex([d for d in DOW if d in heatmap_data.index])
        fig = px.imshow(heatmap_data, aspect="auto", color_continuous_scale="RdYlGn_r",
                        title="Accident Frequency Heatmap: Day × Hour",
                        labels=dict(x="Hour of Day", y="Day of Week", color="Accidents"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", height=380)
        st.plotly_chart(fig, use_container_width=True)

    elif eda_choice == "Risk Score Distributions":
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(dff, x="traffic_density", y="risk_score",
                         color="traffic_density", color_discrete_map=TRAFFIC_COLOR,
                         title="Risk Score by Traffic Density", points="outliers")
            fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(dff, x="city", y="risk_score", color="traffic_density",
                         color_discrete_map=TRAFFIC_COLOR,
                         title="Risk Score by City & Density")
            fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117",
                              paper_bgcolor="#0e1117", xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.violin(dff, x="road_type", y="risk_score", color="traffic_density",
                        color_discrete_map=TRAFFIC_COLOR, box=True,
                        title="Risk Score Violin Plot by Road Type & Traffic Density")
        fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117",
                          paper_bgcolor="#0e1117", height=430)
        st.plotly_chart(fig, use_container_width=True)

    elif eda_choice == "Correlation Heatmap":
        # BUG FIX: festival column is non-numeric → use is_festival flag instead
        num_cols = ["hour","lanes","traffic_signal","temperature","vehicles_involved",
                    "casualties","is_peak_hour","is_weekend","risk_score","is_festival"]
        available = [c for c in num_cols if c in dff.columns]
        corr = dff[available].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Feature Correlation Matrix", aspect="auto", zmin=-1, zmax=1)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", height=520)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(dff.sample(min(3000, len(dff)), random_state=42),
                          x="risk_score", y="casualties",
                          color="traffic_density", color_discrete_map=TRAFFIC_COLOR,
                          opacity=0.6, trendline="ols",
                          title="Risk Score vs Casualties (with Trendline)")
        fig2.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
        st.plotly_chart(fig2, use_container_width=True)

    else:  
        col1, col2 = st.columns(2)
        with col1:
            sev = dff.groupby(["accident_severity","traffic_density"]).size().reset_index(name="Count")
            fig = px.bar(sev, x="accident_severity", y="Count", color="traffic_density",
                         color_discrete_map=TRAFFIC_COLOR, barmode="group",
                         title="Accident Severity vs Traffic Density")
            fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(dff.sample(min(3000, len(dff)), random_state=42),
                             x="vehicles_involved", y="casualties",
                             color="accident_severity",
                             color_discrete_sequence=["#ff4b4b","#ffa500","#00cc66"],
                             opacity=0.6, title="Vehicles Involved vs Casualties by Severity")
            fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)

        sev_city = dff.groupby(["city","accident_severity"]).size().reset_index(name="Count")
        fig = px.bar(sev_city, x="city", y="Count", color="accident_severity",
                     color_discrete_sequence=["#ff4b4b","#ffa500","#00cc66"],
                     barmode="stack", title="Severity Distribution by City")
        fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117",
                          paper_bgcolor="#0e1117", xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)



with tabs[3]:
    st.markdown('<p class="section-header">🤖 Machine Learning — Traffic Density Prediction</p>', unsafe_allow_html=True)

    # Model comparison table
    st.markdown("### 📊 Model Comparison")
    comp_data = []
    for name, res in model_results.items():
        comp_data.append({
            "Model":              name,
            "Train Accuracy":     round(res["train_acc"], 4),
            "Test Accuracy":      round(res["accuracy"], 4),
            "CV Score (5-fold)":  round(res["cv_score"], 4),
            "Diagnosis":          res["diagnosis"],
        })
    comp_df = pd.DataFrame(comp_data).sort_values("Test Accuracy", ascending=False)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(comp_df, x="Model", y=["Train Accuracy","Test Accuracy","CV Score (5-fold)"],
                     barmode="group", title="Model Accuracy Comparison",
                     color_discrete_sequence=["#43e97b","#667eea","#764ba2"])
        fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117",
                          paper_bgcolor="#0e1117", yaxis_range=[0.3, 1.0])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sel_res = model_results[active_model]
        cm      = sel_res["cm"]
        classes = le_target.classes_
        fig = px.imshow(cm, x=classes, y=classes, text_auto=True,
                        color_continuous_scale="Blues",
                        title=f"Confusion Matrix — {active_model}")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### 📋 Classification Report — {active_model}")
    report      = sel_res["report"]
    report_rows = []
    for cls in le_target.classes_:
        if cls in report:
            r = report[cls]
            report_rows.append({
                "Class":     cls.capitalize(),
                "Precision": round(r["precision"], 3),
                "Recall":    round(r["recall"], 3),
                "F1-Score":  round(r["f1-score"], 3),
                "Support":   int(r["support"]),
            })
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

    st.markdown("### 🏆 Top Feature Importances (Gradient Boosting)")
    top_fi = feat_imp.head(15).copy()
    fig = px.bar(top_fi, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="viridis",
                 title="Top 15 Feature Importances")
    fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117",
                      paper_bgcolor="#0e1117",
                      yaxis=dict(autorange="reversed"), height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📉 Per-Class Performance Metrics")
    fig = go.Figure()
    colors_metrics = ["#667eea", "#f64f59", "#43e97b"]
    for m_name, color in zip(["precision","recall","f1-score"], colors_metrics):
        vals = [report[cls][m_name] for cls in le_target.classes_ if cls in report]
        fig.add_trace(go.Bar(x=list(le_target.classes_), y=vals,
                             name=m_name.capitalize(), marker_color=color))
    fig.update_layout(template="plotly_dark", barmode="group",
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      title=f"Per-Class Metrics — {active_model}", height=380)
    st.plotly_chart(fig, use_container_width=True)



with tabs[4]:
    st.markdown('<p class="section-header">🔮 Predict Traffic Density by Location & Conditions</p>',
                unsafe_allow_html=True)
    st.markdown("Fill in the details below to predict the expected traffic density at any location in India.")

    with st.form("traffic_prediction_form"):
        col_p1, col_p2, col_p3 = st.columns(3)

        with col_p1:
            st.markdown("#### 📍 Location")
            pred_city   = st.selectbox("City", sorted(CITY_COORDS.keys()))
            pred_road   = st.selectbox("Road Type", ["highway","urban","rural"])
            pred_lanes  = st.slider("Number of Lanes", 1, 8, 4)
            pred_signal = st.selectbox("Traffic Signal Present", ["Yes","No"])
            pred_signal_val = 1 if pred_signal == "Yes" else 0
            pred_latitude = st.number_input('Latitude')
            pred_longitude = st.number_input("Longitude")

        with col_p2:
            st.markdown("#### 🌤️ Conditions")
            pred_weather = st.selectbox("Weather",    ["clear","fog","rain"])
            pred_vis     = st.selectbox("Visibility", ["high","medium","low"])
            pred_temp    = st.slider("Temperature (°C)", 5, 48, 28)
            pred_cause   = st.selectbox("Primary Cause",
                ["weather","distraction","overspeeding","drunk driving","poor road"])

        with col_p3:
            st.markdown("#### 🕐 Temporal")
            pred_hour = st.slider("Hour of Day", 0, 23, 9)
            pred_dow  = st.selectbox("Day of Week",
                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            pred_weekend  = 1 if pred_dow in ["Saturday","Sunday"] else 0
            pred_peak     = st.selectbox("Is Peak Hour?", ["Yes","No"])
            pred_peak_val = 1 if pred_peak == "Yes" else 0
            pred_month    = st.slider("Month", 1, 12, 6)

        st.markdown("#### 🚘 Incident Details")
        c1, c2, c3,c4 = st.columns(4)
        with c1:
            pred_vehicles  = st.slider("Vehicles Involved", 1, 20, 3)
        with c2:
            pred_casualties = st.slider("Casualties", 0, 30, 1)
        with c3:
            pred_risk = st.slider("Risk Score", 0.0, 1.0, 0.45, 0.05)
        with c4:
            pred_accident = st.selectbox('Accident Serverity',['major','minor','fatal'])

        predict_btn = st.form_submit_button("🔮 Predict Traffic Density", use_container_width=True)

    if predict_btn:
        input_row = {
            "hour":             pred_hour,
            "is_weekend":       pred_weekend,
            "lanes":            pred_lanes,
            "traffic_signal":   pred_signal_val,
            "temperature":      pred_temp,
            "vehicles_involved":pred_vehicles,
            "casualties":       pred_casualties,
            "is_peak_hour":     pred_peak_val,
            "risk_score":       pred_risk,
            "month":            pred_month,
            "road_type":        pred_road,
            "weather":          pred_weather,
            "visibility":       pred_vis,
            "cause":            pred_cause,
            "day_of_week":      pred_dow,
            "city":             pred_city,

        }
        model_pipeline = trained_models[active_model]
        try:
            ct = model_pipeline.named_steps["preprocessor"]
            all_expected_cols = numeric_features + categorical_features
        except Exception:
            all_expected_cols = list(input_row.keys())

        pred_df = pd.DataFrame([input_row])
        for col in all_expected_cols:
            if col not in pred_df.columns:
                pred_df[col] = np.nan
        pred_df = pred_df[all_expected_cols]

        pred_class_idx = model_pipeline.predict(pred_df)[0]
        pred_label     = le_target.inverse_transform([pred_class_idx])[0]
        pred_proba     = model_pipeline.predict_proba(pred_df)[0]

        st.session_state.prediction_data = {
            "label":      pred_label,
            "proba":      pred_proba,
            "city":       pred_city,
            "road":       pred_road,
            "weather":    pred_weather,
            "hour":       pred_hour,
            "dow":        pred_dow,
            "peak":       pred_peak,
            "risk":       pred_risk,
            "vehicles":   pred_vehicles,
            "casualties": pred_casualties,
            "model":      active_model,
        }

    if st.session_state.prediction_data:
        result     = st.session_state.prediction_data
        pred_label = result["label"]
        pred_proba = result["proba"]
        emoji      = TRAFFIC_EMOJI.get(pred_label, "❓")
        css_class  = f"pred-{pred_label}"

        st.markdown(f"""
        <div class="{css_class}" style="margin-top:20px;">
            <h1 style="margin:0;font-size:48px;">{emoji}</h1>
            <h2>Predicted Traffic Density: <b>{pred_label.upper()}</b></h2>
            <p>Model: {result["model"]} | City: {result["city"]}</p>
        </div>
        """, unsafe_allow_html=True)
        

        st.markdown(' ')
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            proba_df = pd.DataFrame({"Class": le_target.classes_, "Probability": pred_proba})
            fig = px.bar(proba_df, x="Class", y="Probability",
                         color="Class", color_discrete_map=TRAFFIC_COLOR,
                         text_auto=".3f", title="Class Probabilities")
            fig.update_layout(template="plotly_dark", yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        with col_r2:
            lat, lon = CITY_COORDS[result["city"]]
            color = ("#ff4b4b" if pred_label == "high"
                     else "#ffa500" if pred_label == "medium"
                     else "#00cc66")
            m2 = folium.Map(location=[lat, lon], zoom_start=12, tiles="cartodb positron")
            folium.CircleMarker(location=[lat, lon], radius=22,
                                color=color, fill=True, fill_color=color,
                                fill_opacity=0.8).add_to(m2)
            st_folium(m2, height=320, key="prediction_map")

        best_prob  = float(np.max(pred_proba))
        conf_level = "High" if best_prob > 0.7 else ("Medium" if best_prob > 0.5 else "Low")
        st.markdown(f"""
        <div class="insight-box">
            🎯 <b>Prediction Confidence:</b> {conf_level} ({best_prob:.1%})
        </div>
        """, unsafe_allow_html=True)
