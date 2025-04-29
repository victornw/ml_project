import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models import Word2Vec
import pickle


def processar_dataset(df, texto_col='Email Text', target_col='Email Type', vector_size=200, window=6, min_count=2, pkl_path='modelo_phishing.pkl'):
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df = df.rename(columns={texto_col: 'texto'})
    df['texto'].replace('empty', np.nan, inplace=True)
    df = df.dropna(subset=['texto'])
    df = df.drop_duplicates(subset='texto', keep='first')

    label_encoder = LabelEncoder()
    df[target_col] = label_encoder.fit_transform(df[target_col])

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
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = remove_accents(text)
        text = remove_stopwords(text)
        return text.strip()

    df['texto'] = df['texto'].apply(preprocess_text)
    df['tokens'] = df['texto'].apply(lambda x: x.lower().split())

    model_w2v = Word2Vec(
        sentences=df['tokens'].tolist(),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        workers=4
    )

    def vetor_medio(texto, model):
        palavras = texto.split()
        vetores = [model.wv[p] for p in palavras if p in model.wv]
        if vetores:
            return np.mean(vetores, axis=0)
        else:
            return np.zeros(model.vector_size)

    vetores = df['texto'].apply(lambda x: vetor_medio(x, model_w2v))
    matriz = np.vstack(vetores.values)

    # Criar DataFrame com vetores e manter a coluna target
    df_w2v = pd.DataFrame(
        matriz, columns=[f'w2v_{i}' for i in range(model_w2v.vector_size)])

    # Garantir que a coluna target está presente e é a última coluna
    df_w2v[target_col] = df[target_col].values

    # Reordenar colunas para deixar a target como última
    colunas = [col for col in df_w2v.columns if col !=
               target_col] + [target_col]
    df_tratado = df_w2v[colunas]

    # Salvar componentes separadamente
    with open('w2v_model.pkl', 'wb') as f:
        pickle.dump(model_w2v, f)

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Salvar dataset tratado para uso no treinamento
    df_tratado.to_csv('dados_RN.csv', index=False)

    return df_tratado, model_w2v


if __name__ == "__main__":
    df = pd.read_csv("Phishing_Email.csv", encoding='latin-1')
    df_tratado, model_w2v = processar_dataset(df)
    print(df_tratado.head())
    print(model_w2v)
