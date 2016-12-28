import numpy

def histeq(image, bins):
	max = numpy.max(image)
	min = numpy.min(image)
	d = (max - min) / bins

	mask = image < min
	x = min	
	for i in xrange(bins):
		x += d
		mask = (image <= x) & ~mask
		print numpy.sum(numpy.multiply(image, mask))

#equalizeHistogram([[1, 3.4, 2],[3, 4, 0.9]], 2)