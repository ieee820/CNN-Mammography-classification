import numpy

def getComponents(bin):
	#c = 4 # connectivity
	sz = bin.shape
	comp = numpy.zeros_like(bin)
	ctr = 1
	for i in xrange(sz[0]):
		for j in xrange(sz[1]):
			if bin[i,j] == 0:
				comp[i,j] = 0
			else:
				if bin[i,j-1] == 0 or j == 0:
					comp[i,j] = ctr
					ctr += 1
				else:
					comp[i,j] = comp[i,j-1]

	#print bin
	#print comp

	dict = {}
	for i in range(1,sz[0]):
		for j in xrange(sz[1]):
			if not comp[i,j] == 0:
				if not comp[i-1,j] == 0:
					#print('%d is %d' % (comp[i,j], comp[i-1,j]))
					#print dict
					p = dict.get(comp[i-1,j]) if (not dict.get(comp[i-1,j]) is None) else comp[i-1,j]
					q = 0
					while not p == q:
						q = p
						p = dict.get(p) if not dict.get(p) is None else p
					dict[comp[i,j]] = p#comp[i-1,j]

	comp2 = comp.copy()
	
	for i in range(sz[0]):
		for j in xrange(sz[1]):
			if not comp2[i,j] == 0:
				if not dict.get(comp2[i,j]) is None:
					#print('Replacing %d with %d' % (comp2[i,j], dict.get(comp2[i,j])))
					comp2[i,j] = dict.get(comp2[i,j])

	#print comp2

	"""ctr = 0
	for j in xrange(sz[1]):
		#if not bin[0,j] == 0:
		#	bin[0,j] = ctr
		for i in range(0,sz[0]):
			if not comp[i,j] == 0:
				if comp[i-1,j] == 0:
					if not i == 0:
						comp[i, j] = comp[i-1,j]
					else
						comp = 
				else:
					ctr += 1
					comp[i, j] = ctr"""
	#print bin
	return comp2

#def getConnectedComponents(bin):
#	con = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]









