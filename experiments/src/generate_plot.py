#!/usr/bin/python
# coding: utf-8
import sys, getopt

def main(argv):
	inputfiles = {}
	outputfile = 'gnuplot.in'
	inputargs = ("cpu_ntt","gpu_ntt","cpu_intt","gpu_intt")
	try:
		opts, args = getopt.getopt(argv,"ha:b:c:d:e:",["cpu_ntt=","gpu_ntt=","cpu_intt=","gpu_intt=","date="])
	except getopt.GetoptError:
		print 'generate_plot.py <' + "> <".join(("help",)+inputargs)+'>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'generate_plot.py <' + "> <".join(("help",)+inputargs)+'>'
			sys.exit()
		elif opt in ("--"+x for x in inputargs):
			inputfiles[opt] = arg
		elif opt in ("--date"):
			for i in inputargs:
				inputfiles[i] = "%s_%s.dat" % (i,arg)
	print 'Input file is "', str(inputfiles)
	print 'Output file is "', outputfile

	gnuplot_in = []

	gnuplot_in.append("set encoding utf8")
	gnuplot_in.append("set terminal postscript")
	gnuplot_in.append("set logscale xy 2")
	gnuplot_in.append("set output \"gnuplot.ps\"")
	gnuplot_in.append("set title \"Tempo necessário para a multiplicação de polinômios de grau arbitrário em escala logarítmica\"")
	gnuplot_in.append("set xlabel \"Grau\"");
	gnuplot_in.append("set ylabel \"Tempo(ms)\"");

	plot = "plot "
	for i in inputfiles.keys():
		plot = plot + "\"%s\" using 1:2 title \"%s\" with lp," % (inputfiles[i],i)
	gnuplot_in.append(plot)

	f = open(outputfile,"w+")
	f.write("\n".join(gnuplot_in))
	f.close()

if __name__ == "__main__":
   main(sys.argv[1:])