# -*- coding: utf-8 -*-
"""predict glasses type of petients with Decision tree method.
   @ excerpted from <Machine Learning in Action> Oct 28th,2018

   while "TypeError: 'dict_keys' object does not support indexing"
   modify d.keys()[] with list(d.keys())[]
"""
from trees import createTree
from treePlotter import createPlot

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = createTree(lenses,lensesLabels)
createPlot(lensesTree)