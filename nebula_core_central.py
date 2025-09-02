import os
import json
import re
import random
import unicodedata
from math import sqrt, factorial, log
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util

# ---------------- Flask App ----------------
app = Flask(__name__)

# ---------------- Caminho base ----------------
BASE_PATH = "bigdata"

# ---------------- Modelo ----------------
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------- Stopwords (palavras a ignorar) ----------------
STOPWORDS = {
    "a", "o", "e", "de", "do", "da", "em", "no", "na", "para", "um", "uma",
    "os", "as", "por", "que", "com", "se", "ao", "dos", "das", "nos", "nas",
    "sobre", "à", "às"
}

# ---------------- Carrega similares do JSON ----------------
SIMILARES_PATH = "similares.json"
if os.path.exists(SIMILARES_PATH):
    with open(SIMILARES_PATH, "r", encoding="utf-8") as f:
        SIMILARES = json.load(f)
else:
    SIMILARES = {}

# ---------------- Dicionários ----------------
NUMEROS = {
    "zero":0, "um":1, "uma":1, "dois":2, "duas":2, "três":3, "quatro":4,
    "cinco":5, "seis":6, "sete":7, "oito":8, "nove":9, "dez":10
}

OPERADORES = {
    "mais":"+",
    "menos":"-",
    "vezes":"*",
    "x":"*",
    "multiplicado por":"*",
    "dividido por":"/",
    "sobre":"/",
    "mod":"%",
    "módulo":"%"
}

palavras_slot1 = [
    "estudo", "matemática", "fisica", "química", "história", "geografia",
    "português", "literatura", "biologia", "escola", "prova", "enem",
    "vestibular", "faculdade", "ciência", "livro", "apostila", "exercício",
    "curso", "professor", "universidade", "educação", "redação"
]

# ---------------- Funções auxiliares ----------------
def normalizar_palavra(txt: str) -> str:
    nfkd = unicodedata.normalize("NFKD", txt.lower())
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def extrair_palavras_chave(texto: str):
    """Remove stopwords e retorna só palavras relevantes"""
    palavras = re.findall(r'\w+', normalizar_palavra(texto))
    return [p for p in palavras if p not in STOPWORDS and len(p) > 2]

def converte_palavras_para_numeros(texto):
    for palavra, numero in NUMEROS.items():
        texto = re.sub(r'\b' + palavra + r'\b', str(numero), texto, flags=re.IGNORECASE)
    return texto

def converte_operadores_em_simbolos(texto):
    for palavra, simbolo in OPERADORES.items():
        texto = re.sub(r'\b' + palavra + r'\b', simbolo, texto, flags=re.IGNORECASE)
    return texto

def calcula_expressao(mensagem):
    texto = mensagem.lower()
    if "raiz quadrada de" in texto:
        numero = re.findall(r'raiz quadrada de (\d+)', texto)
        if numero:
            return f"💻 Resultado: {sqrt(int(numero[0]))}"
    if "fatorial de" in texto:
        numero = re.findall(r'fatorial de (\d+)', texto)
        if numero:
            return f"💻 Resultado: {factorial(int(numero[0]))}"
    if "elevado a" in texto:
        partes = re.findall(r'(\d+)\s*elevado a\s*(\d+)', texto)
        if partes:
            base, expoente = map(int, partes[0])
            return f"💻 Resultado: {base ** expoente}"
    if "log de" in texto:
        numero = re.findall(r'log de (\d+)', texto)
        if numero:
            return f"💻 Resultado: {log(int(numero[0]))}"
    texto = converte_palavras_para_numeros(texto)
    texto = converte_operadores_em_simbolos(texto)
    if re.match(r'^[0-9+\-*/().%\s]+$', texto):
        try:
            resultado = eval(texto)
            return f"💻 Resultado: {resultado}"
        except Exception as e:
            return f"Erro ao calcular: {e}"
    return None

# ---------------- Carrega base RAG ----------------
def carregar_base():
    dados = []
    for slot in range(1, 5):
        slot_path = os.path.join(BASE_PATH, f"slot{slot}", "versionamento1")
        if os.path.exists(slot_path):
            for arquivo in os.listdir(slot_path):
                if arquivo.endswith(".json"):
                    with open(os.path.join(slot_path, arquivo), "r", encoding="utf-8") as f:
                        try:
                            perguntas_respostas = json.load(f)
                            for item in perguntas_respostas:
                                pergunta = item.get("pergunta", "")
                                resposta = item.get("resposta", "")
                                embedding = MODEL.encode(pergunta, convert_to_tensor=True)
                                dados.append({"pergunta": pergunta, "resposta": resposta, "embedding": embedding})
                        except Exception as e:
                            print(f"Erro ao carregar {arquivo}: {e}")
    print(f"[INFO] Base de conhecimento carregada: {len(dados)} perguntas")
    return dados

BASE_CONHECIMENTO = carregar_base()

# ---------------- Função para traduzir a resposta ----------------
def traduzir_resposta(resposta: str) -> str:
    def substituir(match):
        palavra = match.group(0).lower()
        alternativas = SIMILARES.get(palavra, [palavra])
        return random.choice(alternativas)
    return re.sub(r'\b\w+\b', substituir, resposta, flags=re.IGNORECASE)

# ---------------- Função de resposta com palavras-chave ----------------
def procurar_resposta(mensagem, user_ip):
    calculo = calcula_expressao(mensagem.replace(" ", ""))
    if calculo:
        print(f"[CALCULO] IP: {user_ip} | Mensagem: {mensagem} | Resultado: {calculo}")
        return calculo

    # extrai palavras-chave
    palavras = extrair_palavras_chave(mensagem)
    if not palavras:
        return "Desculpe, não entendi sua pergunta."

    # gera embedding para cada palavra
    embeddings_palavras = [MODEL.encode(p, convert_to_tensor=True) for p in palavras]

    melhor_score = -1
    melhor_resposta = "Desculpe, não encontrei uma resposta para isso."

    for item in BASE_CONHECIMENTO:
        score_total = 0
        for emb in embeddings_palavras:
            score_total += util.cos_sim(emb, item["embedding"]).item()
        score_medio = score_total / len(embeddings_palavras)

        if score_medio > melhor_score:
            melhor_score = score_medio
            melhor_resposta = item["resposta"]

    if any(p in mensagem.lower() for p in palavras_slot1):
        melhor_resposta += " 💡 (Esta resposta está no slot 1, chance: 0% - 35%)"

    melhor_resposta = traduzir_resposta(melhor_resposta)

    print(f"[RESPOSTA SELECIONADA] IP: {user_ip} | Score: {melhor_score} | Resposta: {melhor_resposta}")
    return melhor_resposta

# ---------------- Rotas Flask ----------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/mensagem', methods=['POST'])
def mensagem():
    data = request.get_json()
    mensagem_usuario = data.get("mensagem")
    user_ip = request.remote_addr

    if not mensagem_usuario:
        return jsonify({"erro": "Nenhuma mensagem recebida"}), 400

    resposta = procurar_resposta(mensagem_usuario, user_ip)
    return jsonify({"resposta": resposta}), 200

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
