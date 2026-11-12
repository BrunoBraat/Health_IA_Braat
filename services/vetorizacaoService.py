from sklearn.feature_extraction.text import TfidfVectorizer
from services.datasetService import dataset_completo
from sklearn.preprocessing import LabelEncoder

# 1 - Chamar o framework que tem o TF-IDF
# 2 - Instaciar o modelo de vetorização
# 3 - Pegar os dados e vetorizar

#vetorizador
def vetorizador():
    """
    Funcao que cria um vetorizador TF-IDF ajustado aos dados de entrada
    
    Returns:
        tfidf: TfidfVectorizer ajustado
    """
    tfidf = TfidfVectorizer()
    df = dataset_completo()
    x = df["sintomas"].astype(str)
    tfidf.fit(x)
    return tfidf

def encoder():
    """
    Funcao que cria um LabelEncoder ajustado aos diagnosticos
    
    Returns:
        label_encoder: LabelEncoder ajustado
    """
    label_encoder = LabelEncoder()
    df = dataset_completo()
    y = df["diagnostico"].astype(str)
    label_encoder.fit(y)
    return label_encoder

#vetorizador vetorizando os dados
#retorno dados vetorizados
def vetorizacao():

    tfidf = TfidfVectorizer()

    df = dataset_completo()
    x = df["sintomas"].astype(str)

    tfidf.fit(x)

    x_tfidf = tfidf.transform(x)

    return x_tfidf


def encode_Y():
    """
    Função para codificar a variavel alvo (Y) usando o Label Encoding.

    Return
    Y_encoded: array
        Array com os rotulos codificados

    """
    label_encoder = LabelEncoder()
    df = dataset_completo()
    y = df["diagnostico"].astype(str)

    y_encoded = label_encoder.fit_transform(y)

    return y_encoded
