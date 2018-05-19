import pathlib
DIR=str(pathlib.Path(__file__).resolve().parent)

import pandas as pd
from collections import Counter

df = pd.read_csv(DIR + '/situacao_do_cliente.csv')
X_df = df[['recencia','frequencia', 'semanas_de_inscricao']]
Y_df = df['situacao']

Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values 

porcentagem_de_treino = 0.8
porcentagem_de_teste = 0.1

tamanho_de_treino = porcentagem_de_treino * len(Y)
tamanho_de_teste = porcentagem_de_teste * len(Y)
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

treino_dados = X[:int(tamanho_de_treino)]
treino_marcacoes = Y[:int(tamanho_de_treino)]

fim_de_treino = tamanho_de_treino + tamanho_de_teste

teste_dados = X[int(tamanho_de_treino):int(fim_de_treino)]
teste_marcacoes = Y[int(tamanho_de_treino):int(fim_de_treino)]

dados_reais = X[int(fim_de_treino):]
marcacoes_reais = Y[int(fim_de_treino):]


def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
	modelo.fit(treino_dados, treino_marcacoes)

	resultado = modelo.predict(teste_dados)

	acertos = resultado == teste_marcacoes

	total_de_acertos = sum(acertos)
	total_de_elementos = len(teste_dados)

	taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

	msg = "Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_de_acerto)

	print(msg)
	return taxa_de_acerto

def teste_real(modelo, dados_reais, marcacoes_reais):
	resultado = modelo.predict(dados_reais)
	acertos = resultado == marcacoes_reais

	total_de_acertos = sum(acertos)
	total_de_elementos = len(marcacoes_reais)

	taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

	msg = "Taxa de acerto do vencedor entre os tres algoritmos no mundo real: {0}".format(taxa_de_acerto)
	print(msg)

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn import svm
modeloSVM = svm.SVC(gamma=0.001, C=100)
resultadoSVM = fit_and_predict("SVM", modeloSVM, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)


if resultadoMultinomial > resultadoAdaBoost and resultadoMultinomial > resultadoSVM:
	modelo_vencedor = modeloMultinomial
else:
	if resultadoAdaBoost > resultadoSVM:
		modelo_vencedor = modeloAdaBoost
	else:
		modelo_vencedor = modeloSVM


teste_real(modelo_vencedor, dados_reais, marcacoes_reais)

acerto_base = max(Counter(marcacoes_reais).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(marcacoes_reais)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

total_de_elementos = len(dados_reais)
print("Total de teste: %d" % total_de_elementos)