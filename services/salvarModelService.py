from services.vetorizacaoService import vetorizador, encoder
from services.treinamentoService import treinar_modelo
import pickle
import os

#Salvar o vetorizador
#Salvar o ENcoder (Label Y)
#Salvar o modelo treinado

def salvar_vetorizador():
    """
    Salvar o vetorizador como arquivo no diretorio model

    """
    vetorizador_criado = vetorizador()
    #Salvar o vetorizador
    caminho_arquivo = 'models/vetorizador_HealthIA.pkl'
    #if caso nao tenha o diretorio, criar
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok = True)
    with open(caminho_arquivo,"wb") as f:
        pickle.dump(vetorizador_criado, f)
        print("Vetorizador salvo com sucesso!")


def salvar_enconderY():
    """
    Salvar o encoder Y como arquivo no diretorio model

    """
    encoderY_criado = encoder()
    #Salvar o encoderU
    caminho_arquivo = 'models/encoderY_HealthIA.pkl'
    #if caso nao tenha o diretorio, criar
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok = True)
    with open(caminho_arquivo,"wb") as f:
        pickle.dump(encoderY_criado, f)
        print("Encoder Y salvo com sucesso!")


def salvar_modelo():
    """
    Salvar o modelo treinado como arquivo no diretorio model
    
    """
    HealthIA = treinar_modelo()
    #Salvar o modelo
    caminho_arquivo = 'models/modelo_HealthIA.json'
    #if caso nao tenha o diretorio, criar
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok = True)
    
    HealthIA.save_model(caminho_arquivo)
    print("Modelo salvo com sucesso!")