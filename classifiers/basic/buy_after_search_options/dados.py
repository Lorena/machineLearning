import csv

def carregar_buscas():
    X = [];
    Y = [];
    arquivo = open('busca.csv', 'r')
    leitor = csv.reader(arquivo)
    next(leitor)
    for home, busca, logado, comprou in leitor:
        dado = [int(home), busca, int(logado)]
        X.append(dado)
        Y.append(int(comprou))

    return X, Y
