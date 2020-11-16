STUDENTIDS = 500395897_500710654

$(STUDENTIDS).zip: algorithm.py report/report.pdf
	mkdir -p $(STUDENTIDS)/code/algorithm $(STUDENTIDS)/code/data
	cp $< $(STUDENTIDS)/code/algorithm
	cp $(word 2,$^) $(STUDENTIDS)
	cd $(STUDENTIDS) && zip -r ../$@ *
	rm -r $(STUDENTIDS)

report/report.pdf: report/report.tex report/report.sty report/references.bib
	[ -d data ] || (mkdir data && cd data && unzip ../resources/datasets.zip)
	cd report && pdflatex --synctex=1 report.tex && biber report && pdflatex --synctex=1 report.tex && pdflatex --synctex=1 report.tex

lint:
	yapf -d algorithm.py && pycodestyle algorithm.py
