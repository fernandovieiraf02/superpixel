#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
  Nome: bancoImagens.py
  Autor: Hemerson Pistori (pistori@ucdb.br)

  Descricão: Manipular um banco de imagens organizado em pastas para cada classe de objetos ou cenas e salvo em disco

"""

import os
import numpy as np
import cv2

class BancoImagens(object):
    """
       Facilita a manipulação de um banco de imagens gravado em disco. O banco de imagens
       deve ser colocado dentro da pasta 'data'.

    """

    pastaRaiz = '../data/'
    pastaBancoImagens = None
    nomeArquivoArff = None
    classes = None

    def __init__(self, nomeBancoImagens, pastaRaiz = '../data/'):

        self.pastaRaiz = pastaRaiz
        self.pastaBancoImagens = self.pastaRaiz + nomeBancoImagens + "/"
        self.nomeArquivoArff = self.pastaBancoImagens  + nomeBancoImagens + ".arff"

        self.classes = self.lista_pastas()
        self.classes.sort()



    def lista_pastas(self):

       return [name for name in os.listdir(self.pastaBancoImagens)
               if os.path.isdir(os.path.join(self.pastaBancoImagens, name))]




    def imagens_da_classe(self,nomeDaClasse):

        pastaDaClasse = self.pastaBancoImagens + nomeDaClasse + "/"

        imagens = []

        for item in os.listdir(pastaDaClasse):
          if not item.startswith('.'):
            try:
                image = cv2.imread(pastaDaClasse+item)
                imagens.append(image)
            except IOError:
                print 'Não conseguiu abrir imagem '+pastaDaClasse+item

        return imagens






