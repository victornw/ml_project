import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import pickle


def treinar_e_salvar_modelo(df_final, target_col='Email Type', pkl_path='modelo_phishing.pkl'):
    X = df_final.drop(columns=[target_col])
    y = df_final[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=777)

    input_dim = X_train.shape[1]

    model = Sequential()
    model.add(Dense(8, input_dim=input_dim, activation='relu',
              kernel_initializer='he_normal'))
    model.add(Dense(4, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = optimizers.SGD(learning_rate=0.01)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=15, batch_size=32,
              validation_split=0.2, verbose=0)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Salvar apenas o modelo (sem o label_encoder)
    model.save(pkl_path.replace('.pkl', '.h5'))  # Salva como .h5
    
    # Retornar o modelo treinado
    return model


if __name__ == "__main__":
    df = pd.read_csv('dados_RN.csv')
    df_final = df.copy()
    treinar_e_salvar_modelo(
        df_final, target_col='Email Type', pkl_path='modelo_phishing.pkl')
