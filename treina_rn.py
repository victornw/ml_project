import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import pickle

def treinar_e_salvar_modelo(df_final, target_col='Email Type', pkl_path='modelo.pkl'):
    X = df_final.drop(columns=[target_col])
    y = df_final[target_col]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=777)

    input_dim = X_train.shape[1]

    model = Sequential()
    model.add(Dense(8, input_dim=input_dim, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(4, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = optimizers.SGD(learning_rate=0.01)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=0)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in label_encoder.classes_]))

    with open(pkl_path, 'wb') as f:
        pickle.dump((model, label_encoder), f)

if __name__ == "__main__":
    df = pd.read_csv('dados_RN.csv')  
    df_final = df.copy()  
    treinar_e_salvar_modelo(df_final, target_col='Email Type', pkl_path='modelo_phishing.pkl')