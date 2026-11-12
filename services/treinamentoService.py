from services.vetorizacaoService import vetorizacao, encode_Y
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# buscar o modelo XGBoost
from xgboost import XGBClassifier as XGBoost


# passo 1 - buscar os dados - X, Y
def buscar_dados():
    x = vetorizacao()
    y = encode_Y()
    return x, y


# Passo 2 - Separar os dados em treino (70%) e teste (30%)
def separar_dados():
    x, y = buscar_dados()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=42)
    return x_train, x_test, y_train, y_test


# passo 3 - Treinar o modelo com os dados de treino
def treinar_modelo():
    """
    Função para treinar o modelo de machine learning com o lagoritimo XGBoster


    Return
    model: XGBClassifier
        O modelo treinado - que se chama HealthIA.
    """
    # Desempacota os dados numa única chamada
    #x_train, x_test, y_train, y_test = separar_dados()

    x_train = separar_dados()[0]
    y_train = separar_dados()[2]

    # deixando o modelo mais robusto, com mais árvores e uma taxa de aprendizado menor
    HealthIA = XGBoost(n_estimators=150, learning_rate=0.03,
                       max_depth=3, random_state=42)
    HealthIA.fit(x_train, y_train)

    return HealthIA


# Passo 4 - Avaliar o modelo com os dados de teste e calcular métricas
def acuracia_modelo():
    x_train, x_test, y_train, y_test = separar_dados()

    # Treina o modelo
    HealthIA = treinar_modelo()

    # Recebe as previsões do modelo para os dados de teste
    y_pred = HealthIA.predict(x_test)
    acuracia = accuracy_score(y_test, y_pred)

    porcentagem_acuracia = acuracia * 100

    return porcentagem_acuracia
