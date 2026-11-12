
import os
import joblib
import xgboost as xgb
import numpy as np
from services.datasetService import dataset_completo


#Classe com os métodos principais(funções) para executar o modelo


class DiagnosticoIA:
    def __init__(self, caminhoModelo: str = 'models'):
        self.caminho_raiz = caminhoModelo
        self.HealthIA = xgb.XGBClassifier()
        self.HealthIA.load_model(os.path.join(self.caminho_raiz, 'modelo_HealthIA.json'))
        self.vetorizadorfidf = joblib.load(os.path.join(self.caminho_raiz, 'vetorizador_HealthIA.pkl'))
        self.encoderYPronto = joblib.load(os.path.join(self.caminho_raiz, 'encoderY_HealthIA.pkl'))

    def predict_simples(self, sintomas):
        """
        Função para fazer predição do diagnóstico com base nos sintomas
        """
        # Converter lista para string se necessário
        if isinstance(sintomas, list):
            sintomas_string = " ".join(sintomas)
        else:
            sintomas_string = str(sintomas)
            
        # Vetorizar os sintomas
        sintomas_vetorizados = self.vetorizadorfidf.transform([sintomas_string])
            
        # Fazer predição
        predicao = self.HealthIA.predict(sintomas_vetorizados)
        
        # Converter predição numérica para o nome do diagnóstico
        if hasattr(self.encoderYPronto, 'inverse_transform'):
            diagnostico_nome = self.encoderYPronto.inverse_transform(predicao)[0]
        else:
            diagnostico_nome = str(predicao[0])
            
        return diagnostico_nome

