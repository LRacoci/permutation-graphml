all:
	pdflatex projeto_pesquisa
	bibtex projeto_pesquisa
	pdflatex projeto_pesquisa

clean: 
	rm -f *.aux *.bbl *.log *.blg *.dvi *.toc

cleanall: clean
	rm -f projeto_pesquisa.pdf
