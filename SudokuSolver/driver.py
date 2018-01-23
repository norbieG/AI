'''
Sudoku solver using AC3 algorithm and bactracking
The initial state of board is passed from command line and
the solved version is written into the output.txt file
'''

import sys
from itertools import permutations
from collections import defaultdict
import numpy as np
from copy import deepcopy
import pdb
import time

rows = 'ABCDEFGHI'
cols = '123456789'

class csp:
	def __init__(self, init_board):
		
		self.D = self.add_domain(self.get_board(init_board)) #domain
		self.X = self.D.keys() #variables
		self.C = self.get_constraints() #constraints
		self.binary_constr = self.binarise_constraints(self.C) #binary constraints
		self.neighbours = self.get_neighbours()

	def get_board(self, data):
		board = {}
		matrix = []
		
		X = [r+c  for r in rows for c in cols]
		line = [int(digit)for digit in data]
		
		for index in xrange(81):
			board[X[index]] = [line[index]]

		return board

	def add_domain(self,board):
		domain = [1,2,3,4,5,6,7,8,9]
		for key in board.keys():
			if board[key] == [0]:
				board[key] = domain
		return board

	def get_constraints(self):

		row_constr = [tuple([r+c for c in cols]) for r in rows]
		col_constr = [tuple([r+c for r in rows]) for c in cols]
		box_constr = [tuple([r+c for c in bc for r in br]) for br in ['ABC', 'DEF', 'GHI'] for bc in ['123', '456', '789']]

		return row_constr + col_constr + box_constr

	def binarise_constraints(self,constraints):

		return {t for c in constraints for t in tuple(permutations(c,2))}

	def get_neighbours(self):

		neig = defaultdict(list)
		for x in self.X:
			for con in self.C:
				if x in con:
					neig[x]+= [e for e in con if e!=x and e not in neig[x]]
		return neig

	def AC3(self):
		constraints = deepcopy(self.binary_constr)
		queue = constraints

		while len(queue)!= 0:
			(X_i, X_j) = queue.pop()
			if self.revise(X_i, X_j):
				if len(self.D[X_i]) == 0:
					return False
					
				for X_k in self.neighbours[X_i]:
					if X_k!=X_j:
						queue.add((X_k,X_i))
		return True

	def revise(self, X_i, X_j):
		revised = False
		new_domain = [e for e in self.D[X_i]]

		for x in self.D[X_i]:
			for y in self.D[X_j]:
				if x!=y:
					break
			else:
				new_domain.remove(x)
				revised = True
		
		if revised: self.D[X_i] = new_domain

		return revised
			
	def is_complete(self):
		
		for constr in self.binary_constr:
			
			if len(self.D[constr[0]])!=1 or len(self.D[constr[1]])!=1 or  self.D[constr[0]] == self.D[constr[1]]:
				return False

		return True


	def select_unassigned_var(self):

		variables = [ d for d in sorted(self.D, key = lambda x: len(self.D[x])) if len(self.D[d])>1]

		if len(variables)!=0:
			
			return variables[0]

		return 0 
		
start = time.clock()

def backtrack(state):

	csp = deepcopy(state)

	if csp.is_complete():

		print 'solved!'
		print np.array([csp.D[r+c][0] for r in rows for c in cols]).reshape(9,9)

		return csp

	cell = csp.select_unassigned_var()

	for v in csp.D[cell]:

		csp.D[cell] = [v]
		
		if csp.AC3():
			result = backtrack(csp)
			if result!= False:
				return result
		
		csp.D[cell] = state.D[cell]
		csp = state
			

	return False


sudoku = csp(sys.argv[1])

sudoku.AC3()

solution = backtrack(sudoku)

print time.clock() - start


solved = [solution.D[x][0] for x in sorted(solution.X)]
the_file = open('output.txt', 'w')
for item in solved:
  the_file.write("%s" % item)
the_file.write("\n")


