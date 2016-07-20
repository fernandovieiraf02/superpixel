#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
  Nome: extrai_atributos.py
  Autor: Hemerson Pistori (pistori@ucdb.br)
         Alessandro dos Santos Ferreira ( asf2005kn@hotmail.com )

  Descricão: Extrai vários atributos de um banco de imagens organizado com uma subpasta para cada classe do banco e gerar um arquivo arff que pode ser utilizado com o weka
"""

import sys
import os
import cv2
from arff import Arff
from bancoImagens import BancoImagens
from extratores import Extratores

class ExtraiAtributos(object): 

    nomeBancoImagens = None
    nomePastaRaiz = None

    def __init__(self, nomeBancoImagens, nomePastaRaiz):
        self.nomeBancoImagens = nomeBancoImagens
        self.nomePastaRaiz = nomePastaRaiz
        
                            
    def extractAll(self, nomeArquivoArff = None, classes = None, overwrite = True):

        print 'Gerando ARFF para o Banco de Imagens ' + self.nomeBancoImagens + "..."

        bancoImagens = BancoImagens(self.nomeBancoImagens,self.nomePastaRaiz)
        extratores = Extratores()

        print 'Localização do Banco ' + bancoImagens.pastaBancoImagens
        
        nomeArquivoArff = bancoImagens.nomeArquivoArff if nomeArquivoArff is None else bancoImagens.pastaBancoImagens + nomeArquivoArff
        
        if overwrite == False and os.path.isfile(nomeArquivoArff):
            print 'Arquivo ARFF encontrado em ' + nomeArquivoArff
            return 

        if classes is None:
            classes = bancoImagens.classes

            print 'Classes Encontradas'
            print classes

        # Aqui começa a extração de atributos de todas as imagens de cada classe

        dados = []
        nomesAtributos = []
        tiposAtributos = []
        valoresAtributos = []

        for classe in classes:
            imagens = bancoImagens.imagens_da_classe(classe)

            print "Processando %s imagens da classe %s " % (len(imagens),classe)

            for imagem in imagens:
                nomesAtributos, tiposAtributos, valoresAtributos = extratores.extrai_todos(imagem)

                dados.append(valoresAtributos+[classe])

        if len(classes) > 0:
            Arff().cria(nomeArquivoArff, dados, self.nomeBancoImagens, nomesAtributos, tiposAtributos, bancoImagens.classes)

        print 'Arquivo ARFF gerado em ' + nomeArquivoArff

    
    def extractOneFile(self, nomeArquivoArff, nomeImagem):
        
        bancoImagens = BancoImagens(self.nomeBancoImagens,self.nomePastaRaiz)
        extratores = Extratores()
        
        nomeArquivoArff = bancoImagens.pastaBancoImagens + nomeArquivoArff
        
        dados = []
        nomesAtributos = []
        tiposAtributos = []
        valoresAtributos = []
       
        imagem = cv2.imread(bancoImagens.pastaBancoImagens + nomeImagem)
        nomesAtributos, tiposAtributos, valoresAtributos = extratores.extrai_todos(imagem)
        
        dados.append(valoresAtributos+[bancoImagens.classes[0]])
        
        Arff().cria(nomeArquivoArff, dados, self.nomeBancoImagens, nomesAtributos, tiposAtributos, bancoImagens.classes)
