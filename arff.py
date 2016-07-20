#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
  Nome: arff.py
  Autor: Hemerson Pistori (pistori@ucdb.br)

  Descricão: Gera um arquivo arff


"""

import io

class Arff(object):


    def cria(self,nomeArquivo, dados, nomeRelacao, nomesAtributos, tiposAtributos, nomesClasses):

        arquivo = open(nomeArquivo,'wb')

        arquivo.write("%s %s\n\n" % ('@relation ', nomeRelacao))

        for nome, tipo in zip(nomesAtributos,tiposAtributos):
            arquivo.write("%s %s %s\n" % ('@attribute', nome, tipo))


        arquivo.write("%s %s {%s}\n\n" % ('@attribute','classe',', '.join(nomesClasses)))

        arquivo.write('@data\n\n')

        for instancia in dados:

            instancia = map(str, instancia)
            line = ",".join(instancia)
            arquivo.write(line+"\n")

        arquivo.close()



