from services.datasetService import dataset_completo
from services.vetorizacaoService import vetorizacao, encode_Y
from services.treinamentoService import buscar_dados, separar_dados, treinar_modelo, acuracia_modelo
from services.treinamentoMsgService import acuracia_modelo_with_msg
from services.salvarModelService import salvar_vetorizador, salvar_enconderY, salvar_modelo

#Metodos para imprimir
def acuracia_modelo_print():
    acuracia = acuracia_modelo()
    print(f"Acurácia do modelo: {acuracia:.2f}%")

def acuracia_modelo_commsg_print():
    acuracia = acuracia_modelo_with_msg()
    print(f"Acurácia do modelo: {acuracia:.2f}%")

def prints_dataset():
    df = dataset_completo()
    return df

def prints_encode_Y():
    y_encoded = encode_Y()
    print(y_encoded)


def prints_vetorizacao():
    x_tfidf = vetorizacao()
    print(x_tfidf)


#Metodos para salvar o encoder, o modelo e o vetorizador
def salvar_enconderY_healthIA():
    salvar_enconderY()

def salvar_modelo_healthIA():
    salvar_modelo()

def salvar_vetorizador_healthIA():
    salvar_vetorizador()


#Metodo principal
if __name__ == "__main__":
    # prints_vetorizacao()
    # df = prints_dataset()
    # print(df["diagnostico"].unique().sum())
    # prints_encode_Y()

    # Função sem mensagem de acurácia do modelo
    #acuracia_modelo_print()

    # Função com mensagem de acurácia do modelo
    #acuracia_modelo_commsg_print()

    
    #salvar_vetorizador_healthIA()
    #salvar_enconderY_healthIA()
    salvar_modelo_healthIA()