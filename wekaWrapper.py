#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
  Nome: weka.py
  Autor: Alessandro dos Santos Ferreira ( asf2005kn@hotmail.com )

  Descricão: 
"""

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random

#import matplotlib.pyplot as plt
#import weka.plot.classifiers as plcls 

class Weka(object): 

    data = None
    dataDir = None
    classifier = None

    def __init__(self, dataDir = '.'):
        self.dataDir = dataDir 
        
        jvm.start()
        

    # Inicializa dados com conteudo do arquivo arff
    def initData(self, arrfFile):
        loader = Loader(classname="weka.core.converters.ArffLoader")
        print self.dataDir + '/' + arrfFile
        self.data = loader.load_file(self.dataDir + '/' + arrfFile)
        self.data.class_is_last()
        
        print 'Carregando arquivo ' + self.dataDir + '/' + arrfFile
        # print(data)
                     

    # Realiza o treinamento do classificador
    def trainData(self, arrfFile = None, classname="weka.classifiers.trees.J48", options=["-C", "0.3"]):
        if arrfFile is not None:
            self.initData( arrfFile )
            
        if self.data is None:
            return 
            
        print 'Contruindo classificador ' + str(classname) + ' ' + ' '.join(options)
        self.classifier = Classifier(classname=classname, options=options)
        self.classifier.build_classifier(self.data)


    # Realiza a classificacao das instancias de um arquivo arff
    def classify(self, predictFile):
            
        if self.data is None or self.classifier is None:
            return [-1]

        loader = Loader(classname="weka.core.converters.ArffLoader")
        predict_data = loader.load_file(self.dataDir + '/' + predictFile)
        predict_data.class_is_last()
        
        values = str(predict_data.class_attribute)[19:-1].split(',')
        
        classes = []
        
        for index, inst in enumerate(predict_data):
            #pred = self.classifier.classify_instance(inst)
            prediction = self.classifier.distribution_for_instance(inst)
            cl = int(values[prediction.argmax()][7:])
            
            print 'Classe:', cl
            classes.append(cl)

        return classes


    # Realiza uma validação cruzada e mostra os resultados na saída padrão
    def crossValidate(self, arrfFile = None, classname="weka.classifiers.trees.J48", options=["-C", "0.3"]):
        
        if arrfFile is not None:
            self.initData( arrfFile )
            
        if self.data is None:
            return 

        print 'Classificador ' + str(classname) + ' ' + ' '.join(options)
        cls = Classifier(classname=classname, options=options)
        
        evl = Evaluation(self.data)
        evl.crossvalidate_model(cls, self.data, 10, Random(1))

        print(evl.percent_correct)
        print(evl.summary())
        print(evl.class_details())

        #plcls.plot_roc(evl, class_index=[0, 1], wait=True)
