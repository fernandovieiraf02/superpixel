Título: python-extrai-atributos
Autor: Hemerson Pistori
Resumo: Código em python que extrai diversos atributos e gera um arff para ser processado pelo weka. A meta é pode extrair todos os atributos implementados no OpenCV e no scikit-image.

----- Atributos extraídos:

1)Atributos de cor RGB, HSV, Cielab (Mín., Máx., média e Desvp)
2)Descritor de forma, invariante a escala, translação e rotação: 7 momentos de Hu 
3)Atributos de Textura – GLCM (contrastes, dissimilaridades, homogeneidades, asm, energias, correlações)
4)Forma e orientação: HOG
5)Atributos de textura: LPB

----- Dependências

- kubuntu trusty 14.04.2 TLS
- Python 2.7.6 
- scikit-image
- Opencv 2.7

## Dependências Windows

- Instale o [Anaconda](http://continuum.io/downloads) que contém todas depenência, inclusive o Python. Basta fazer o download do arquivo .exe e executa-lo.



----- Como instalar o OpenCV

  Seguir as instruções disponíveis em http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation
  Lí em algum lugar que dá para instalar com o comando abaixo, não testei mas pode funcionar:
  $ sudo apt-get install python-opencv

## Como instalar o OpenCV no Windows
 - [OpenCV-Python](https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html#install-opencv-python-in-windows).
	1. Baixe o [Opencv](https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html#install-opencv-python-in-windows)
	2. Extraia os arquivos no local desejado.
	3. Vá para a pasta opencv/build/python/2.7.
	4. Copie o arquivo cv2.pyd para C:/Python27/lib/site-packeges.
	5. Abra o terminal e digite python para executar o interpretador.
	6. Digite:

    	```(python)
        >>> import cv2
        >>> print cv2.__version__
        ```
    Pronto!




----- Como instalar scikit-image e arff

  $ sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose python-pip python-networkx 
  $ sudo pip install -U scikit-image 

  Em uma das máquinas em que tentei instalar deu um erro que resolvi rodando o comando abaixo antes de executar a linha acima:
  $ sudo apt-get build-dep python-matplotlib


 
----- Como Usar
 
- Crie uma nova pasta dentro da pasta 'data' com o seu banco de imagens (siga a mesma estrutura da pasta de exemplo que está lá, leia o README dela)
- Entre na pasta com o código fonte (que também é 'executável/interpretável' no caso do python)
  $ cd src  
- Execute o código que extrai atributos passando como parâmetro o nomes do seu banco de imagens (apenas o nome, não o caminho)
- $ python ./extraiAtributos.py nome_do_banco_de_imagens
- O arquivo arff será gerado dentro da pasta do seu banco de imagens

  Opcionalmente é possível trabalhar com a pasta original onde está o seu banco de imagens, passando mais um parâmetro para o extraiAtributos.py:

- $ python ./extraiAtributos.py nome_do_banco_de_imagens nome_da_pasta_onde_esta_o_banco_de_imagens


----- Como adicionar novos extratores

- Para extratores que já estão no opencv ou scikit-image
  * Insira o código dentro src/extratores.py (leia as orientações e tente fazer bem parecido com os extratores que já estão lá)

- Para novos extratores
  * Insira a implementação do novo extrator dentro de um novo código na pasta src/novos-extratores
  * Insira a chamada para este novo extrator dentro de src/extratores.py
