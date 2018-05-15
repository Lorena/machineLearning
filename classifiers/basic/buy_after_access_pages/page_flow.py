import pathlib
DIR=str(pathlib.Path(__file__).resolve().parent)

import csv

def low_page_access():
    X = []
    Y = []

    arquivo = open(DIR + '/acesso.csv', 'r')
    leitor = csv.reader(arquivo)

    next(leitor)

    for home, como_funciona, contato, comprou in leitor:
        dado = [int(home), int(como_funciona), int(contato)]
        X.append(dado)
        Y.append(int(comprou))
    return X, Y