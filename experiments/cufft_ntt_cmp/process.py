#!/usr/bin/python
result = {}
from polynomial import Polynomial
import ast

def process(x):
	CRTSize = int(x[0])
	degree = int(x[1])
	a = Polynomial()
	b = Polynomial()
	c = Polynomial()
	a.coef = list(ast.literal_eval(x[2]))
	b.coef = list(ast.literal_eval(x[3]))
	c.coef = list(ast.literal_eval(x[4]))

	while c.coef[-1] == 0:
		c.coef.pop()

	ab = a*b

	count = len([x for i,x in enumerate(ab.coef) if x != c.coef[i]])

	# Verifies if a*b == c
	return [CRTSize, degree, count]

def reduction(x):
	CRTSize = x[0]
	degree = x[1]
	test = x[2]

	if CRTSize not in result.keys():
		result[CRTSize] = {}
	if degree not in result[CRTSize].keys():
		result[CRTSize][degree] = 0

	result[CRTSize][degree] = test

import sys
assert len(sys.argv) >= 2
file = open(sys.argv[1])
split = lambda A,n: [A[i:i+n] for i in range(0, len(A), n)]

data = split(file.readlines(),5)

from multiprocessing import Pool
p = Pool()
map_result = p.map(process, data)

for x in map_result:
	reduction(x)

import json

with open("resultado.json", 'w+') as outfile:
	json.dump(result,outfile)
