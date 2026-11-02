"""PhishGuard — interface Streamlit. Rodar com: streamlit run app.py"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dados import carregar, matriz

try:  # parte do Gabriel: metricas.py com calcular_metricas(vp, fp, fn, vn) -> dict
    from metricas import calcular_metricas
except ImportError:
    calcular_metricas = None

CSV_DEMO = Path(__file__).with_name("dados_phishguard.csv")

st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ PhishGuard")
st.caption("Avaliador de detector de mensagens de phishing — matriz de confusão e efeito do limiar.")

# --- Importação dos dados ---
st.sidebar.header("Dados")
enviado = st.sidebar.file_uploader("CSV (mensagem_id, classe_real, probabilidade_phishing)", type="csv")
fonte = enviado or (CSV_DEMO if CSV_DEMO.exists() else None)
if fonte is None:
    st.info("Envie um CSV pela barra lateral ou coloque `dados_phishguard.csv` ao lado do app.py.")
    st.stop()

try:
    df = carregar(fonte)
except ValueError as erro:
    st.error(str(erro))
    st.stop()

st.sidebar.success(f"{enviado.name if enviado else CSV_DEMO.name}: {len(df)} mensagens")
if df.attrs["descartadas"]:
    st.sidebar.warning(f"{df.attrs['descartadas']} linha(s) inválida(s) descartada(s).")

limiar = st.sidebar.slider("Limiar de classificação", 0.0, 1.0, 0.5, 0.01,
                           help="Probabilidade mínima para marcar a mensagem como phishing.")

# --- Base ---
st.subheader("Base carregada")
if df.attrs["so_teste"]:
    st.caption("Avaliação apenas no conjunto de teste (coluna `conjunto`).")
phishing = int(df["classe_real"].sum())
c1, c2, c3 = st.columns(3)
c1.metric("Total", len(df))
c2.metric("Phishing", phishing, f"{phishing / len(df):.0%}", delta_color="off")
c3.metric("Legítimas", len(df) - phishing, f"{1 - phishing / len(df):.0%}", delta_color="off")
with st.expander("Ver dados"):
    st.dataframe(df, width="stretch", hide_index=True)

# --- Matriz de confusão ---
m = matriz(df, limiar)
st.subheader(f"Matriz de confusão — limiar {limiar:.2f}")
col_graf, col_txt = st.columns([3, 2])
fig = px.imshow(
    [[m["VP"], m["FN"]], [m["FP"], m["VN"]]],
    x=["Previsto: phishing", "Previsto: legítimo"],
    y=["Real: phishing", "Real: legítimo"],
    text_auto=True, color_continuous_scale="Blues",
)
fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
fig.update_traces(textfont_size=22)
col_graf.plotly_chart(fig, width="stretch")
col_txt.markdown(
    f"""
- **VP = {m['VP']}** — phishing identificado corretamente
- **FN = {m['FN']}** — phishing que passou como legítimo
- **FP = {m['FP']}** — mensagem legítima bloqueada como phishing
- **VN = {m['VN']}** — mensagem legítima identificada corretamente
"""
)

# --- Distribuição das probabilidades ---
st.subheader("Distribuição das probabilidades por classe")
hist = px.histogram(
    df.assign(Classe=df["classe_real"].map({1: "Phishing", 0: "Legítima"})),
    x="probabilidade_phishing", color="Classe", nbins=40, barmode="overlay", opacity=0.6,
    color_discrete_map={"Phishing": "#d62728", "Legítima": "#1f77b4"},
    labels={"probabilidade_phishing": "Probabilidade prevista de phishing"},
)
hist.add_vline(x=limiar, line_dash="dash", annotation_text=f"limiar {limiar:.2f}")
hist.update_layout(yaxis_title="Mensagens", margin=dict(t=30, b=10))
st.plotly_chart(hist, width="stretch")
st.caption("À direita da linha tudo é marcado como phishing: barras azuis ali são FP, barras vermelhas à esquerda são FN.")

# --- Métricas (parte do Gabriel) ---
st.subheader("Métricas")
if calcular_metricas:
    for col, (nome, valor) in zip(st.columns(4), calcular_metricas(m["VP"], m["FP"], m["FN"], m["VN"]).items()):
        col.metric(nome, f"{valor:.3f}")
else:
    st.info("Aguardando `metricas.py` com `calcular_metricas(vp, fp, fn, vn)`.")

# --- Comparação dos limiares da demonstração ---
st.subheader("Comparação de limiares")
cenarios = pd.DataFrame([{"Limiar": f"{l:.2f}", **matriz(df, l)} for l in (0.30, 0.50, 0.80)])
col_tab, col_bar = st.columns([2, 3])
col_tab.dataframe(cenarios, width="stretch", hide_index=True)
barras = px.bar(
    cenarios.melt(id_vars="Limiar", var_name="Tipo", value_name="Mensagens"),
    x="Limiar", y="Mensagens", color="Tipo", barmode="group", text_auto=True,
    color_discrete_map={"VP": "#2ca02c", "FP": "#ff7f0e", "FN": "#d62728", "VN": "#1f77b4"},
)
barras.update_layout(margin=dict(t=10, b=10))
col_bar.plotly_chart(barras, width="stretch")

st.subheader("Erros conforme o limiar varia")
curva = pd.DataFrame([{"Limiar": l / 100, **matriz(df, l / 100)} for l in range(101)])
linhas = px.line(
    curva, x="Limiar", y=["FP", "FN"],
    color_discrete_map={"FP": "#ff7f0e", "FN": "#d62728"},
    labels={"value": "Mensagens", "variable": "Erro"},
)
linhas.add_vline(x=limiar, line_dash="dash")
linhas.update_layout(margin=dict(t=10, b=10))
st.plotly_chart(linhas, width="stretch")
st.caption("FP: mensagem legítima bloqueada. FN: phishing que chega ao usuário. Subir o limiar troca um erro pelo outro.")
