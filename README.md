# 🩺 Health_IA_Braat — API de Predição de Diagnóstico Médico

API REST construída com **FastAPI** e **Machine Learning (XGBoost + TF-IDF NLP)** para predição de diagnósticos médicos e apoio à triagem a partir de sintomas relatados em texto livre.

> Projeto prático desenvolvido durante a formação **Mastertech Machine Learning** (IT Valley School), simulando um fluxo completo de ciência de dados e engenharia de ML em produção: ingestão de dados, vetorização de texto (NLP), codificação de classes, treinamento, serialização de artefatos e disponibilização via API.

---

## 📋 Índice

- [Visão Geral da Arquitetura](#-visão-geral-da-arquitetura)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação e Configuração](#-instalação-e-configuração)
- [Como Executar](#-como-executar)
- [Documentação e Exemplos de Endpoints](#-documentação-e-exemplos-de-endpoints)
- [Autor](#-autor)

---

## 🧠 Visão Geral da Arquitetura

O pipeline do **Health_IA_Braat** é dividido em duas etapas principais:

1. **Pipeline de Treinamento e Serialização:**
   - **Vetorização (NLP):** Transformação das descrições textuais de sintomas em matrizes numéricas utilizando **TF-IDF** (`TfidfVectorizer`).
   - **Codificação de Rótulos:** Mapeamento das classes diagnósticas para valores inteiros utilizando `LabelEncoder`.
   - **Modelagem:** Divisão estratificada dos dados em treino ($75\%$) e teste ($25\%$) e treinamento com classificador **XGBoost** (`XGBClassifier`), alcançando ~77%–78% de acurácia no conjunto de validação.
   - **Persistência:** Exportação do vetorizador, do encoder e do modelo treinado em formato `.pkl` (`models/`).

2. **Inferência em Tempo Real (API FastAPI):**
   - Carregamento dos artefatos serializados (`vetorizador_HealthIA.pkl`, `encoderY_HealthIA.pkl` e `modelo_HealthIA.pkl`).
   - Recebimento dos sintomas via requisição HTTP, transformação com o vetorizador treinado, predição via XGBoost e decodificação reversa (`inverse_transform`) para retornar o nome do diagnóstico previsto.

---

## 🚀 Tecnologias Utilizadas

- **Linguagem:** Python 3.13+
- **Framework Web / API:** FastAPI & Uvicorn
- **Machine Learning & NLP:** Scikit-Learn (`TfidfVectorizer`, `LabelEncoder`, `train_test_split`), XGBoost (`XGBClassifier`)
- **Manipulação de Dados:** Pandas & NumPy
- **Serialização:** Pickle

---

## 📂 Estrutura do Projeto

```text
Health_IA_Braat/
├── models/                         # Artefatos serializados do pipeline de ML
│   ├── vetorizador_HealthIA.pkl    # Vetorizador TF-IDF treinado
│   ├── encoderY_HealthIA.pkl       # LabelEncoder das classes de diagnóstico
│   └── modelo_HealthIA.pkl         # Modelo XGBoost treinado
├── routers/                        # Rotas e endpoints da API FastAPI
│   └── ...                         # Endpoints de sintomas e diagnóstico
├── services/                       # Camada de regras de negócio e ML
│   ├── datasetService.py           # Leitura e preparação do dataset
│   ├── vetorizacaoService.py       # Vetorização TF-IDF e codificação do alvo (Y)
│   ├── treinamentoService.py       # Split de dados, treino XGBoost e avaliação
│   ├── salvarModelService.py       # Exportação dos arquivos .pkl
│   └── ...                         # Serviço de execução de diagnóstico
├── leitura.py                      # Script orquestrador de treino, métricas e exportação
├── main.py                         # Ponto de entrada da aplicação FastAPI
├── requirements.txt                # Dependências do projeto
└── .gitignore                      # Arquivos e diretórios ignorados pelo Git
```

---

## ⚙️ Instalação e Configuração

### 1. Clonar o repositório
```bash
git clone https://github.com/BrunoBraat/Health_IA_Braat.git
cd Health_IA_Braat
```

### 2. Criar e ativar o ambiente virtual (`venv`)

**No Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**No Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

---

## ▶️ Como Executar

### 1. (Opcional) Treinar e gerar os artefatos `.pkl`
Caso queira retreinar o modelo e atualizar os arquivos na pasta `models/`:
```bash
python leitura.py
```

### 2. Subir o servidor da API FastAPI
Na raiz do projeto, execute o servidor com o Uvicorn:
```bash
uvicorn main:app --reload
```

A API estará disponível em: `http://127.0.0.1:8000`

---

## 📡 Documentação e Exemplos de Endpoints

O FastAPI gera automaticamente a interface interativa Swagger UI para testes diretamente pelo navegador:

- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Exemplos de Requisições

#### 🔹 Listar sintomas de exemplo (`GET /sintomas`)
```bash
curl -X GET "http://127.0.0.1:8000/sintomas"
```

**Resposta esperada (`200 OK`):**
```json
{
  "sintomas": [
    "febre",
    "tosse",
    "dor de cabeça",
    "fadiga",
    "dor muscular",
    "perda de olfato",
    "perda de paladar"
  ]
}
```

#### 🔹 Realizar predição de diagnóstico (`POST /diagnostico`)
*(Nota: ajuste a rota e a chave do payload conforme o nome definido no seu router de diagnóstico).*

```bash
curl -X POST "http://127.0.0.1:8000/diagnostico" \
     -H "Content-Type: application/json" \
     -d '{"sintomas": "febre alta, dor no corpo, fadiga intensa e tosse seca"}'
```

**Resposta esperada (`200 OK`):**
```json
{
  "diagnostico_previsto": "Gripe / Influenza"
}
```

---

## ⚠️ Aviso Legal

Este projeto tem caráter **estritamente educacional e de portfólio**, demonstrando a aplicação de técnicas de Processamento de Linguagem Natural (NLP) e Machine Learning em APIs REST. Os resultados gerados pelo modelo **não substituem** avaliação, diagnóstico ou orientação médica profissional.

---

## 👨‍💻 Autor

Desenvolvido por **Bruno Braat**  
🔗 GitHub: [@BrunoBraat](https://github.com/BrunoBraat)