STUDENTIDS = 500395897_500710654

$(STUDENTIDS).zip: requirements.txt report/report.pdf
	[ -d data ] || (mkdir data && cd data && unzip ../resources/datasets.zip)
	mkdir -p $(STUDENTIDS)/code/algorithm $(STUDENTIDS)/code/data
	cp $< algorithm.py $(STUDENTIDS)/code/algorithm
	cp $(word 2,$^) $(STUDENTIDS)
	cd $(STUDENTIDS) && zip -r ../$@ *
	rm -r $(STUDENTIDS)

requirements.txt: algorithm.py setup.py setup.cfg pyproject.toml
	[ -d venv ] || virtualenv venv
	venv/bin/pip install -U pip
	venv/bin/pip install .
	venv/bin/pip freeze >$@
	rm -r venv

report/report.pdf: report/report.tex report/report.sty report/references.bib
	cd report && pdflatex --synctex=1 report.tex && biber report && pdflatex --synctex=1 report.tex && pdflatex --synctex=1 report.tex

lint:
	yapf -d algorithm.py && pycodestyle algorithm.py
