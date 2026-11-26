import os
import re
import json
import time
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai.errors import ServerError, ClientError

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FinSight | Analyse des Réclamations Bancaires",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_html(content):
    clean = "\n".join(line.lstrip() for line in content.splitlines())
    st.markdown(clean, unsafe_allow_html=True)

# --- EXECUTIVE FINTECH CSS ---
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide ONLY Deploy button, 3-dots menu & footer (keeps sidebar toggle visible!) */
    [data-testid="stAppDeployButton"], .stDeployButton, #MainMenu, footer {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="collapsedControl"], [data-testid="stExpandSidebarButton"] {
        visibility: visible !important;
        color: #0F172A !important;
    }

    /* Main canvas background */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    /* Clean light sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] input {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
    }

    /* Page header banner */
    .top-header {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .top-header-kicker {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #1D4ED8;
        margin-bottom: 6px;
    }
    .top-header-title {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 6px 0;
        line-height: 1.25;
    }
    .top-header-sub {
        font-size: 14px;
        color: #64748B;
        margin: 0;
    }

    /* Elevated cards */
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .card-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 8px;
    }
    .card-title {
        font-size: 16px;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid #F1F5F9;
    }

    /* Prediction highlight box */
    .pred-box {
        background-color: #EFF6FF;
        border-left: 4px solid #1D4ED8;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 18px;
    }
    .pred-dept {
        font-size: 18px;
        font-weight: 700;
        color: #1E3A8A;
        margin-top: 2px;
    }

    /* Custom horizontal confidence bars */
    .bar-row { margin-bottom: 12px; }
    .bar-meta {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        color: #334155;
        margin-bottom: 4px;
    }
    .bar-meta strong { color: #0F172A; }
    .bar-track {
        width: 100%;
        height: 8px;
        background-color: #F1F5F9;
        border-radius: 999px;
        overflow: hidden;
    }
    .bar-fill { height: 100%; border-radius: 999px; }

    /* Urgency badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 8px;
    }
    .badge-high { background-color: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }
    .badge-med  { background-color: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
    .badge-low  { background-color: #F0FDF4; color: #15803D; border: 1px solid #BBF7D0; }

    /* Email draft box */
    .email-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 16px 18px;
        font-size: 13.5px;
        line-height: 1.65;
        color: #1E293B;
        white-space: pre-wrap;
    }

    /* KPI cards */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: left;
    }
    .kpi-value { font-size: 28px; font-weight: 700; color: #0F172A; margin-top: 4px; }
    .kpi-sub { font-size: 12px; color: #64748B; margin-top: 2px; }

    /* Buttons */
    div.stButton > button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 13.5px;
        transition: all 0.15s ease;
    }
    div.stButton > button[kind="primary"] {
        background-color: #0F172A;
        border: 1px solid #0F172A;
        color: #FFFFFF;
        padding: 10px 18px;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8;
        border-color: #1D4ED8;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #E2E8F0; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 500; color: #64748B; padding: 10px 16px; }
    .stTabs [aria-selected="true"] { color: #0F172A !important; font-weight: 600 !important; }

    /* Sidebar status pill */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
</style>
""")

# --- LOAD SAVED NLP ARTIFACTS ---
@st.cache_resource
def load_nlp_artifacts():
    model = joblib.load("best_nlp_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'x{2,}', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_svm_scores(model, vec):
    raw_scores = model.decision_function(vec)[0]
    exp_scores = np.exp(raw_scores - np.max(raw_scores))
    probs = exp_scores / exp_scores.sum()
    df_scores = pd.DataFrame({
        "Département": model.classes_,
        "Score": np.round(probs * 100, 1)
    }).sort_values(by="Score", ascending=False)
    return df_scores

def call_gemini_prescriptive(api_key, raw_complaint, predicted_dept):
    client = genai.Client(api_key=api_key)

    preferred = [
        "gemini-flash-lite-latest",
        "gemini-2.5-flash-lite",
        "gemini-3-flash-preview",
        "gemini-flash-latest"
    ]
    available = [
        m.name.replace("models/", "")
        for m in client.models.list()
        if "flash" in m.name.lower() and "tts" not in m.name.lower() and "image" not in m.name.lower()
    ]
    models_to_try = [m for m in preferred if m in available] + [m for m in available if m not in preferred]

    prompt = f"""
    Tu es un assistant IA expert en conformité et service client bancaire.
    Une réclamation client a été automatiquement classée par notre modèle NLP dans le département : "{predicted_dept}".

    Réclamation brute du client (en anglais) :
    "{raw_complaint}"

    Retourne UNIQUEMENT un objet JSON valide (sans texte avant ni après) contenant exactement ces 5 clés :
    {{
      "resume_manager": "Résumé synthétique en français pour le manager (2 phrases maximum).",
      "cause_racine": "Cause racine identifiée en français (1 phrase précise).",
      "niveau_urgence": "Faible" ou "Moyen" ou "Élevé",
      "justification_urgence": "Courte justification du niveau d'urgence en français (1 phrase).",
      "reponse_client": "Proposition d'e-mail de réponse empathique et professionnelle en anglais, prête à être envoyée."
    }}
    """

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            text = response.text.strip()
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'^```\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            parsed = json.loads(text)
            return parsed, model_name, None
        except json.JSONDecodeError:
            return None, model_name, response.text
        except (ServerError, ClientError):
            time.sleep(1)
            continue

    raise Exception("Aucun modèle Gemini Flash n'est disponible pour le moment. Réessayez dans quelques secondes.")

# --- SIDEBAR ---
with st.sidebar:
    render_html("""
    <div style="padding: 6px 0 18px 0; border-bottom: 1px solid #E2E8F0;">
        <div style="font-size: 18px; font-weight: 700; letter-spacing: -0.02em; color: #0F172A;">
            FINSIGHT ANALYTICS
        </div>
        <div style="font-size: 12px; color: #64748B; margin-top: 4px;">
            Master IASDO 
        </div>
    </div>
    """)

    render_html("<div style='height: 16px;'></div>")
    render_html("<div style='font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #64748B; margin-bottom: 6px;'>Configuration API</div>")

    api_key = st.text_input(
        "Clé API Google Gemini",
        type="password",
        placeholder="AIza...",
        label_visibility="collapsed"
    )

    if api_key:
        render_html("<div style='font-size: 12px; color: #15803D; margin-top: 6px;'><span class='status-dot' style='background:#22C55E;'></span>Service IA Générative actif</div>")
    else:
        render_html("<div style='font-size: 12px; color: #B45309; margin-top: 6px;'><span class='status-dot' style='background:#F59E0B;'></span>En attente de clé API</div>")

    render_html("<div style='height: 24px;'></div>")
    render_html("""
    <div style="border-top: 1px solid #E2E8F0; padding-top: 18px; font-size: 12.5px; line-height: 1.65;">
        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #64748B; margin-bottom: 10px;">
            Architecture du Pipeline
        </div>
        <div style="margin-bottom: 10px;">
            <strong style="color: #0F172A;">01. Prétraitement NLP</strong><br>
            <span style="color: #475569;">Nettoyage Regex, suppression des masques XXXX, vectorisation TF-IDF (1-2 grammes).</span>
        </div>
        <div style="margin-bottom: 10px;">
            <strong style="color: #0F172A;">02. Modèle Prédictif</strong><br>
            <span style="color: #475569;">Linear SVM équilibré (Accuracy 89,73 %, F1-Macro 0,8515).</span>
        </div>
        <div>
            <strong style="color: #0F172A;">03. Module Prescriptif</strong><br>
            <span style="color: #475569;">Synthèse managériale, diagnostic et rédaction via Google Gemini.</span>
        </div>
    </div>
    """)

# --- HEADER ---
render_html("""
<div class="top-header">
    <div class="top-header-kicker">NLP & Intelligence Artificielle Générative — Secteur Bancaire</div>
    <div class="top-header-title">Système Intelligent de Classification et de Résolution des Réclamations</div>
    <div class="top-header-sub">Routage automatisé par apprentissage supervisé (CFPB Dataset) couplé à l'aide à la décision prescriptive par LLM.</div>
</div>
""")

tab1, tab2 = st.tabs(["Démonstration en Direct", "Analyse Exploratoire & Performance"])

# ================= TAB 1: LIVE DEMO =================
with tab1:
    try:
        nlp_model, tfidf = load_nlp_artifacts()
    except FileNotFoundError:
        st.error("Fichiers `best_nlp_model.pkl` ou `tfidf_vectorizer.pkl` introuvables dans le répertoire courant.")
        st.stop()

    examples = {
        "Recouvrement (Debt Collection)": (
            "Wakefield Associates keeps contacting me by SMS about a medical debt even though "
            "I sent a written debt verification request by certified mail over 45 days ago and never received a response."
        ),
        "Crédit Immobilier (Mortgage)": (
            "My mortgage servicer increased my monthly escrow payment by $400 without explanation "
            "and applied my last two payments to suspense instead of principal and interest."
        ),
        "Carte de Crédit / Fraude (Credit Card)": (
            "Someone opened two fraudulent credit card accounts under my name and address. "
            "I reported the identity theft to the bank three weeks ago, yet late fees are still being charged."
        )
    }

    if "complaint_input" not in st.session_state:
        st.session_state.complaint_input = examples["Crédit Immobilier (Mortgage)"]

    render_html("<div class='card-label'>Jeux de test rapides</div>")
    cols_btn = st.columns(3)
    for idx, (label, text) in enumerate(examples.items()):
        if cols_btn[idx].button(label, use_container_width=True):
            st.session_state.complaint_input = text

    raw_complaint = st.text_area(
        "Contenu brut de la réclamation client (anglais) :",
        value=st.session_state.complaint_input,
        height=130
    )

    col_run, _ = st.columns([1, 2])
    run_btn = col_run.button("Exécuter l'analyse complète", type="primary", use_container_width=True)

    if run_btn:
        if not raw_complaint.strip():
            st.warning("Veuillez saisir une réclamation avant de lancer l'analyse.")
        else:
            render_html("<div style='height: 12px;'></div>")
            col_nlp, col_gen = st.columns([1, 1.35], gap="large")

            # --- STAGE 1: CLASSICAL NLP ---
            with col_nlp:
                cleaned = clean_text(raw_complaint)
                vec = tfidf.transform([cleaned])
                predicted_dept = nlp_model.predict(vec)[0]
                scores_df = get_svm_scores(nlp_model, vec)

                bars_html = ""
                for _, row in scores_df.iterrows():
                    dept = html.escape(str(row["Département"]))
                    score = row["Score"]
                    bar_color = "#1D4ED8" if row["Département"] == predicted_dept else "#CBD5E1"
                    bars_html += f"""
                    <div class="bar-row">
                        <div class="bar-meta">
                            <span>{dept}</span>
                            <strong>{score:.1f} %</strong>
                        </div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: {max(score, 1.5)}%; background-color: {bar_color};"></div>
                        </div>
                    </div>
                    """

                render_html(f"""
                <div class="card">
                    <div class="card-label">Étape 1 — Classification NLP (Prédictif)</div>
                    <div class="card-title">Routage Automatique vers le Département</div>
                    <div class="pred-box">
                        <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #3B82F6; font-weight: 600;">
                            Département prédit
                        </div>
                        <div class="pred-dept">{html.escape(predicted_dept)}</div>
                    </div>
                    <div class="card-label" style="margin-top: 14px;">Distribution des scores relatifs (Linear SVM)</div>
                    {bars_html}
                </div>
                """)

                with st.expander("Inspecter le texte prétraité (Étape 3 — Nettoyage Regex)"):
                    st.code(cleaned, language=None)

            # --- STAGE 2: GENERATIVE AI ---
            with col_gen:
                if not api_key:
                    render_html("""
                    <div class="card">
                        <div class="card-label">Étape 2 — IA Générative (Prescriptif)</div>
                        <div class="card-title">Synthèse Managériale & Résolution</div>
                        <div style="color: #64748B; font-size: 14px; padding: 12px 0;">
                            Renseignez votre clé API Google Gemini dans le panneau latéral gauche pour générer la synthèse, le diagnostic de cause racine et la proposition de réponse client.
                        </div>
                    </div>
                    """)
                else:
                    with st.spinner("Génération de l'analyse prescriptive en cours..."):
                        try:
                            parsed, used_model, raw_fallback = call_gemini_prescriptive(
                                api_key, raw_complaint, predicted_dept
                            )

                            if parsed:
                                urgence = str(parsed.get("niveau_urgence", "Moyen")).strip()
                                u_lower = urgence.lower()
                                if "élev" in u_lower or "elev" in u_lower or "haut" in u_lower:
                                    badge_cls = "badge-high"
                                elif "faibl" in u_lower:
                                    badge_cls = "badge-low"
                                else:
                                    badge_cls = "badge-med"

                                resume = html.escape(str(parsed.get("resume_manager", "")))
                                cause = html.escape(str(parsed.get("cause_racine", "")))
                                justif = html.escape(str(parsed.get("justification_urgence", "")))
                                email_txt = html.escape(str(parsed.get("reponse_client", "")))

                                render_html(f"""
                                <div class="card">
                                    <div class="card-label">Étape 2 — IA Générative (Prescriptif)</div>
                                    <div class="card-title">Aide à la Décision & Résolution Client</div>

                                    <div style="margin-bottom: 16px;">
                                        <span class="badge {badge_cls}">Urgence : {html.escape(urgence)}</span>
                                        <div style="font-size: 13.5px; color: #475569;">{justif}</div>
                                    </div>

                                    <div style="margin-bottom: 16px;">
                                        <div class="card-label">Résumé pour le manager</div>
                                        <div style="font-size: 14px; color: #0F172A; line-height: 1.6;">{resume}</div>
                                    </div>

                                    <div style="margin-bottom: 18px; padding: 12px 14px; background-color: #F8FAFC; border-left: 3px solid #64748B; border-radius: 4px;">
                                        <div class="card-label" style="margin-bottom: 4px;">Cause racine identifiée</div>
                                        <div style="font-size: 13.5px; color: #1E293B;">{cause}</div>
                                    </div>

                                    <div>
                                        <div class="card-label">Proposition de réponse client (Prête à l'envoi)</div>
                                        <div class="email-box">{email_txt}</div>
                                    </div>

                                    <div style="font-size: 11.5px; color: #94A3B8; margin-top: 14px; text-align: right;">
                                        Moteur génératif : {html.escape(used_model)}
                                    </div>
                                </div>
                                """)
                            else:
                                render_html("<div class='card'><div class='card-label'>Étape 2 — IA Générative (Prescriptif)</div>")
                                st.markdown(raw_fallback)
                                render_html("</div>")

                        except Exception as e:
                            st.error(f"Erreur lors de l'appel à l'API Gemini : {e}")

# ================= TAB 2: DATA SCIENCE RESULTS =================
with tab2:
    render_html("<div style='height: 8px;'></div>")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_html("""
        <div class="kpi-card">
            <div class="card-label">Échantillon Stratifié</div>
            <div class="kpi-value">19 969</div>
            <div class="kpi-sub">Issu de 486 416 textes valides</div>
        </div>
        """)
    with k2:
        render_html("""
        <div class="kpi-card">
            <div class="card-label">Accuracy (Test)</div>
            <div class="kpi-value">89,73 %</div>
            <div class="kpi-sub">Linear SVM (class_weight=balanced)</div>
        </div>
        """)
    with k3:
        render_html("""
        <div class="kpi-card">
            <div class="card-label">F1-Score (Weighted)</div>
            <div class="kpi-value">0,8971</div>
            <div class="kpi-sub">Évalué sur 3 994 réclamations</div>
        </div>
        """)
    with k4:
        render_html("""
        <div class="kpi-card">
            <div class="card-label">F1-Score (Macro)</div>
            <div class="kpi-value">0,8515</div>
            <div class="kpi-sub">+0,15 pts vs Naive Bayes (0,7050)</div>
        </div>
        """)

    render_html("<div style='height: 20px;'></div>")

    render_html("""
    <div class="card">
        <div class="card-label">Étape 4 — Analyse Exploratoire des Données (EDA)</div>
        <div class="card-title">Distribution Stratifiée, Évolution Temporelle et Structure Textuelle</div>
    </div>
    """)
    if os.path.exists("eda_charts.png"):
        st.image("eda_charts.png", use_container_width=True)
    else:
        st.info("Placez `eda_charts.png` dans le répertoire du projet pour afficher ce graphique.")

    render_html("<div style='height: 16px;'></div>")

    render_html("""
    <div class="card">
        <div class="card-label">Étape 5 — Évaluation du Modèle Supervisé</div>
        <div class="card-title">Matrice de Confusion — Linear SVM sur le Jeu de Test</div>
    </div>
    """)
    if os.path.exists("confusion_matrix.png"):
        col_img, _ = st.columns([1.4, 1])
        col_img.image("confusion_matrix.png", use_container_width=True)
    else:
        st.info("Placez `confusion_matrix.png` dans le répertoire du projet pour afficher la matrice de confusion.")