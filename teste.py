import random

frase = "teste outro teste"

variaveis = {
    "teste": ["teste1", "teste2"],
    "outro": ["diferente", "alternativo"]
}

# Função para substituir palavras por alternativas
def substituir_frase(frase, dicionario):
    palavras = frase.split()
    nova_frase = []
    for p in palavras:
        if p in dicionario:
            nova_frase.append(random.choice(dicionario[p]))
        else:
            nova_frase.append(p)
    return " ".join(nova_frase)

resultado = substituir_frase(frase, variaveis)
print(resultado)
