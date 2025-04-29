import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from trata_rn import processar_dataset
from treina_rn import treinar_e_salvar_modelo
import os
from datetime import datetime
import re
import unicodedata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Configuração da página
st.set_page_config(page_title="Phishing Email Classifier", layout="wide")

# Título e descrição
st.title("📧 Phishing Email Classifier")
st.markdown("""
    Este aplicativo classifica e-mails como phishing/spam ou legítimos. 
    Você pode verificar um e-mail ou contribuir com novos exemplos para melhorar o modelo.
""")

# Sidebar com informações
st.sidebar.header("Sobre")
st.sidebar.info("""
    **Como funciona:**
    1. Insira o texto de um e-mail suspeito
    2. O modelo irá classificar como phishing ou legítimo
    3. Você pode ajudar a melhorar o modelo reportando novos exemplos
""")

# Carregar modelo e processadores


@st.cache_resource
def load_phishing_model():
    try:
        # Carregar modelo Keras
        model = load_model('modelo_phishing.h5')

        # Carregar modelo Word2Vec
        with open('w2v_model.pkl', 'rb') as f:
            w2v_model = pickle.load(f)

        # Carregar LabelEncoder (se necessário)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        return model, w2v_model, label_encoder
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None, None, None


model, w2v_model, label_encoder = load_phishing_model()

# Função para pré-processar texto do usuário


def preprocess_user_input(text, w2v_model):
    # Copiar exatamente o mesmo pré-processamento do trata_rn.py
    def remove_accents(input_str):
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return ''.join([c for c in nfkd_form if not unicodedata.category(c) == 'Mn'])

    def remove_stopwords(text):
        words = text.split()
        return " ".join([word for word in words if word not in ENGLISH_STOP_WORDS])

    def preprocess_text(text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'https?://\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove pontuação
        text = re.sub(r'\s+', ' ', text)  # Remove espaços extras
        text = remove_accents(text)  # Remove acentos
        text = remove_stopwords(text)  # Remove stopwords
        return text.strip()

    def vetor_medio(texto, model):
        # IMPORTANTE: Usar a mesma tokenização do tratamento original
        palavras = texto.lower().split()  # Exatamente como no trata_rn.py
        vetores = [model.wv[p] for p in palavras if p in model.wv]
        if vetores:
            return np.mean(vetores, axis=0)
        else:
            return np.zeros(model.vector_size)

    # Aplicar o mesmo pipeline completo
    processed_text = preprocess_text(text)

    # Tokenização idêntica à usada no treinamento
    # Igual ao df['tokens'] do trata_rn.py
    tokens = processed_text.lower().split()

    # Garantir que o Word2Vec foi treinado com esses tokens
    vector = vetor_medio(processed_text, w2v_model)

    return vector.reshape(1, -1)


# Seção de classificação de e-mail
st.header("🔍 Classificar E-mail")
email_text = st.text_area(
    "Cole o conteúdo do e-mail suspeito aqui:", height=200)

# Seção de classificação de e-mail - Corrigida
if st.button("Classificar E-mail"):
    if not email_text.strip():
        st.warning("Por favor, insira o texto do e-mail para classificação.")
    elif model is None or w2v_model is None:
        st.error(
            "Modelo não carregado corretamente. Verifique os arquivos do modelo.")
    else:
        try:
            input_vector = preprocess_user_input(email_text, w2v_model)
            prediction = model.predict(input_vector)

            # Correção principal aqui:
            pred_class = "Legítimo" if prediction[0][0] > 0.5 else "Phishing/Spam"
            confidence = (
                1 - prediction[0][0]) if pred_class == "Phishing/Spam" else prediction[0][0]

            if pred_class == "Phishing/Spam":
                st.error(
                    f"Resultado: {pred_class} (confiança: {confidence:.2%})")
                st.markdown("**Cuidado!** Este e-mail parece ser malicioso.")
            else:
                st.success(
                    f"Resultado: {pred_class} (confiança: {confidence:.2%})")
                st.markdown("Este e-mail parece ser legítimo.")

        except Exception as e:
            st.error(f"Erro durante a classificação: {e}")

# Seção para reportar novos e-mails
st.header("📩 Reportar Novo E-mail")
st.markdown("Ajude a melhorar o modelo reportando e-mails phishing ou legítimos.")

with st.form("report_form"):
    new_email = st.text_area("Texto do e-mail:", height=150)
    email_type = st.radio("Tipo do e-mail:", ("Phishing/Spam", "Legítimo"))
    reporter_note = st.text_input("Observações (opcional):")
    submitted = st.form_submit_button("Reportar E-mail")

# Seção para reportar novos e-mails - Corrigida
if submitted:
    if not new_email.strip():
        st.warning("Por favor, insira o texto do e-mail.")
    else:
        try:
            new_row = {
                "Email Text": new_email,
                # Correção aqui:
                "Email Type": 0 if email_type == "Phishing/Spam" else 1,
                "Note": reporter_note,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            contributions = pd.concat(
                [contributions, pd.DataFrame([new_row])], ignore_index=True)

            # Salvar
            contributions.to_csv("user_contributions.csv", index=False)
            st.success(
                "Obrigado por sua contribuição! O e-mail foi salvo para futuras atualizações do modelo.")
        except Exception as e:
            st.error(f"Erro ao salvar a contribuição: {e}")

# Seção para retreinar o modelo
st.header("🔄 Atualizar Modelo")
st.markdown("""
    Quando houver contribuições suficientes, você pode retreinar o modelo 
    para incorporar os novos exemplos.
""")

if st.button("Retreinar Modelo com Novos Dados"):
    try:
        # Carregar dados originais e contribuições
        original_data = pd.read_csv("dados_RN.csv", encoding='latin-1')
        try:
            contributions = pd.read_csv("user_contributions.csv")
            if not contributions.empty:
                # Preparar contribuições para mesclar
                contrib_data = contributions[[
                    "Email Text", "Email Type"]].copy()

                # Combinar dados
                combined_data = pd.concat(
                    [original_data, contrib_data], ignore_index=True)

                # Processar e treinar
                with st.spinner("Processando dados e treinando novo modelo..."):
                    # Processar dados
                    df_tratado, w2v_model = processar_dataset(combined_data)

                    # Treinar modelo
                    model = treinar_e_salvar_modelo(df_tratado)

                # Limpar contribuições após treinamento
                os.remove("user_contributions.csv")

                st.success("Modelo atualizado com sucesso!")
                st.balloons()

                # Forçar recarregar os modelos
                load_phishing_model.clear()
                model, w2v_model, label_encoder = load_phishing_model()
            else:
                st.warning(
                    "Nenhuma contribuição encontrada para treinar o modelo.")
        except FileNotFoundError:
            st.warning("Nenhuma contribuição encontrada para treinar o modelo.")
    except Exception as e:
        st.error(f"Erro durante o retreinamento: {e}")

# Seção para mostrar e-mails recentemente reportados
st.header("📊 E-mails Recentemente Reportados")
try:
    contributions = pd.read_csv("user_contributions.csv")
    if not contributions.empty:
        # Mostrar apenas os últimos 10
        recent_contributions = contributions.sort_values(
            "Timestamp", ascending=False).head(10)

        # Converter tipo numérico para legível
        recent_contributions["Tipo"] = recent_contributions["Email Type"].map(
            {1: "Phishing/Spam", 0: "Legítimo"})

        st.dataframe(
            recent_contributions[["Timestamp", "Tipo", "Note"]].rename(columns={
                "Timestamp": "Data/Hora",
                "Note": "Observação"
            }),
            hide_index=True
        )

        # Mostrar estatísticas
        st.subheader("Estatísticas")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Contribuições", len(contributions))
        with col2:
            phishing_count = contributions["Email Type"].sum()
            st.metric("E-mails Phishing Reportados", phishing_count)
    else:
        st.info("Nenhum e-mail reportado ainda. Seja o primeiro a contribuir!")
except FileNotFoundError:
    st.info("Nenhum e-mail reportado ainda. Seja o primeiro a contribuir!")
