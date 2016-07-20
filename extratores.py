#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
  Nome: extratores.py
  Autor: Hemerson Pistori (pistori@ucdb.br)

  Descricão: Define a classe Extratores que faz a ponte com os diversos extratores de
  atributos implementados no OpenCV e no scikit-image. É aqui que devem ser inseridos
  novos códigos para fazer a ponte com novos atributos.

  Modo de Usar:

  - Copie e cole uma das funções que já implementam algum extrator
  - Altere o nome e modifique o código naquilo que for necessário
  - Coloque o nome do extrator, que deve ser igual à função, na lista de extratores construída
    logo no início da classe dentro da variável extratores (não esqueça do "self")
  - Coloque a sigla do extrator na lista criada logo abaixo dos nomes dos extratores

"""

import numpy as np
from skimage import feature, measure, data, color, exposure
import cv2

# Tipo de atributos utilizados pelo Weka
numerico = 'numeric'
nominal = 'nominal'

# Diversos parâmetros utilizados no pré-processamento
cannyMin = 100
cannyMax = 200
glcmNiveis = 256
lbpRaio = 2
nBins = 18

# Controla algumas opções de apresentação de resultados
salvaImagens = False

class Extratores(object):
    """

        Dá acesso aos diversos extratores de atributos implementados no OpenCV e no scikit-image

        Para cada extrator, faz a extração usando um conjunto pré-definido e bem grande de parâmetros

        Acrescente a função para o novo extrator em baixo da última e não esqueça de atualizar as
        variáveis extratores  (lá em baixo, dentro da função extrai_todos)

    """

    imagem = None  # Os extratores trabalharão em cima desta imagem, no formato BGR do OpenCV
    imagemTonsDeCinza = None  # Extratores que dependem de imagens em tons de cinza usarão esta imagem
    imagemBinaria = None  # Extratores que dependem de imagens binárias usarão esta imagem
    imagemBorda = None  # Extratores que dependem de imagens apenas com as bordas detectadas usarão esta imagem
    imagemTamanhoFixo = None # Extratores que dependem de imagens de tamanho fixo usarão esta imagem reescalonada para 128x128
    sequenciaImagens = 1  # Usado para nomear as imagens intermediárias que serão salvas em /tmp

    def momentos_hu(self):
        """
            Calcula os 7 momentos de Hu

        """

        m = measure.moments(self.imagemTonsDeCinza)

        row = m[0, 1] / m[0, 0]
        col = m[1, 0] / m[0, 0]

        mu = measure.moments_central(self.imagemTonsDeCinza,row,col)
        nu = measure.moments_normalized(mu)
        hu = measure.moments_hu(nu)

        valores = list(hu)

        nomes = [m+n for m,n in zip(['hu_'] * len(valores),map(str,range(0,len(valores))))]

        tipos = [numerico] * len(nomes)

        return nomes, tipos, valores


    def estatisticas_cores(self):
        """
         Calcula os valores mínimo, máximo, média e desvio padrão para cada um dos canais nos espaços de
         cores RGB, HSV e CIELab

        """

        nomes = []
        tipos = []
        valores = []

        imagemHSV = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2HSV)
        imagemCIELab = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2LAB)

        b, g, r = cv2.split(self.imagem)
        h, s, v = cv2.split(imagemHSV)
        ciel, ciea, cieb = cv2.split(imagemCIELab)

        nomes = [
            'cor_rmin', 'cor_rmax', 'cor_rmediamedia', 'cor_rdesvio',
            'cor_gmin', 'cor_gmax', 'cor_gmedia', 'cor_gdesvio',
            'cor_bmin', 'cor_bmax', 'cor_bmedia', 'cor_bdesvio',
            'cor_hmin', 'cor_hmax', 'cor_hmedia', 'cor_hdesvio',
            'cor_smin', 'cor_smax', 'cor_smedia', 'cor_sdesvio',
            'cor_vmin', 'cor_vmax', 'cor_vmedia', 'cor_vdesvio',
            'cor_cielmin', 'cor_cielmax', 'cor_cielmedia', 'cor_cieldesvio',
            'cor_cieamin', 'cor_cieamax', 'cor_cieamedia', 'cor_cieadesvio',
            'cor_ciebmin', 'cor_ciebmax', 'cor_ciebmedia', 'cor_ciebdesvio'
        ]

        tipos = [numerico] * len(nomes)

        valores = [
            np.min(r), np.max(r), np.mean(r), np.std(r),
            np.min(g), np.max(g), np.mean(g), np.std(g),
            np.min(b), np.max(b), np.mean(b), np.std(b),
            np.min(h), np.max(h), np.mean(h), np.std(h),
            np.min(s), np.max(s), np.mean(s), np.std(s),
            np.min(v), np.max(v), np.mean(v), np.std(v),
            np.min(ciel), np.max(ciel), np.mean(ciel), np.std(ciel),
            np.min(ciea), np.max(ciea), np.mean(ciea), np.std(ciea),
            np.min(cieb), np.max(cieb), np.mean(cieb), np.std(cieb)
        ]

        return nomes, tipos, valores

    def matriz_coocorrencia(self):
        """

        Extraí atributos de textura baseados em matrizes de coocorrência (GLCM). São utilizadas matrizes 4x4
        nas distäncias 1 e 2 e com ângulos 0, 45 e 90.

        """

        g = feature.greycomatrix(self.imagemTonsDeCinza, [1, 2], [0, np.pi / 4, np.pi / 2], glcmNiveis,normed=True, symmetric=True)

        contrastes = feature.greycoprops(g, 'contrast').tolist()
        dissimilaridades = feature.greycoprops(g, 'dissimilarity').tolist()
        homogeneidades = feature.greycoprops(g, 'homogeneity').tolist()
        asm = feature.greycoprops(g, 'ASM').tolist()
        energias = feature.greycoprops(g, 'energy').tolist()
        correlacoes = feature.greycoprops(g, 'correlation').tolist()

        nomes = [
            'glcm_cont_1_0', 'glcm_cont_1_45', 'glcm_cont_1_90', 'glcm_cont_2_0', 'glcm_cont_2_45', 'glcm_cont_2_90',
            'glcm_diss_1_0', 'glcm_diss_1_45', 'glcm_diss_1_90', 'glcm_diss_2_0', 'glcm_diss_2_45', 'glcm_diss_2_90',
            'glcm_homo_1_0', 'glcm_homo_1_45', 'glcm_homo_1_90', 'glcm_homo_2_0', 'glcm_homo_2_45', 'glcm_homo_2_90',
            'glcm_asm_1_0', 'glcm_asm_1_45', 'glcm_asm_1_90', 'glcm_asm_2_0', 'glcm_asm_2_45', 'glcm_asm_2_90',
            'glcm_ener_1_0', 'glcm_ener_1_45', 'glcm_ener_1_90', 'glcm_ener_2_0', 'glcm_ener_2_45', 'glcm_ener_2_90',
            'glcm_corr_1_0', 'glcm_corr_1_45', 'glcm_corr_1_90', 'glcm_corr_2_0', 'glcm_corr_2_45', 'glcm_corr_2_90',

        ]
        tipos = [numerico] * len(nomes)

        valores = contrastes[0] + contrastes[1] + dissimilaridades[0] + dissimilaridades[1] + homogeneidades[0] + \
                  homogeneidades[1] + asm[0] + asm[1] + energias[0] + energias[1] + correlacoes[0] + correlacoes[1]

        return nomes, tipos, valores


    def hog(self):
        """

        Extraí atributos HOG (Histogramas de Gradientes Orientados)

        """

        valores, hog_image = feature.hog(self.imagemTamanhoFixo, orientations=8, pixels_per_cell=(32, 32),
                    cells_per_block=(1, 1), visualise=True)

        nomes = [m+n for m,n in zip(['hog_'] * len(valores),map(str,range(0,len(valores))))]


        tipos = [numerico] * len(nomes)

        return nomes, tipos, list(valores)



    def lbp(self):
        """

        Extraí atributos LBP (Local Binary Patterns)

        """
        
        lbp = feature.local_binary_pattern(self.imagemTonsDeCinza, 8 * lbpRaio, lbpRaio, 'uniform')
        valores, _ = np.histogram(lbp, normed=True, bins=nBins, range=(0, nBins))
        nomes = [m+n for m,n in zip(['lbp_'] * len(valores),map(str,range(0,len(valores))))]
        tipos = [numerico] * len(nomes)

        return nomes, tipos, list(valores)


    def extrai_todos(self, imagem):

        """

        Chama todos os extratores disponíveis e extrai todos os atributos que eles podem extrair

        :param imagem: Uma imagem do tipo que o OpenCV trabalha

        :return: Todos os nomes, tipos e valores dos atributos extraídos da imagem de entrada

        """

        extratores = [self.momentos_hu, self.estatisticas_cores, self.matriz_coocorrencia, self.hog, self.lbp]

        todosNomesAtributos = []
        todosTiposAtributos = []
        todosValoresAtributos = []

        self.imagem = imagem
        self.imagemTonsDeCinza = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2GRAY)
        self.imagemBorda = cv2.Canny(self.imagemTonsDeCinza, cannyMin, cannyMax)
        ret,self.imagemBinaria = cv2.threshold(self.imagemTonsDeCinza,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.imagemTamanhoFixo = cv2.resize(self.imagemTonsDeCinza, (128, 128))

        if salvaImagens:
            cv2.imwrite("/tmp/img-" + str(self.sequenciaImagens) + "-original.jpg", self.imagem)
            cv2.imwrite("/tmp/img-" + str(self.sequenciaImagens) + "-cinza.jpg", self.imagemTonsDeCinza)
            cv2.imwrite("/tmp/img-" + str(self.sequenciaImagens) + "-borda.jpg", self.imagemBorda)
            cv2.imwrite("/tmp/img-" + str(self.sequenciaImagens) + "-binaria.jpg", self.imagemBinaria)
            cv2.imwrite("/tmp/img-" + str(self.sequenciaImagens) + "-fixo.jpg", self.imagemTamanhoFixo)
            self.sequenciaImagens += 1
  
        for extrator in extratores:

            nomesAtributo, tiposAtributo, valoresAtributo = extrator()

            todosNomesAtributos = todosNomesAtributos + nomesAtributo
            todosTiposAtributos = todosTiposAtributos + tiposAtributo
            todosValoresAtributos = todosValoresAtributos + valoresAtributo


        return todosNomesAtributos, todosTiposAtributos, todosValoresAtributos
