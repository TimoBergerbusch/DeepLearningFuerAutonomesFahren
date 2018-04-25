######################################################################
# 
# makefile 
#  fuer normale Latex-Projekte
#
# Anzupassen sind: GENDIR, MAINTEX und ggf. BIB

# target folder
GENDIR=target/

# primary tex file
MAINTEX=Masterarbeit.tex

# Bibfile
BIB=src/bib/Literatur.bib

# target:
TARGET=$(GENDIR)$(MAINTEX:.tex=.pdf)

# all used latex sources
SOURCES=$(MAINTEX) $(wildcard src/tex/*.tex)

# to trac changes of pictures
PICTURES=$(wildcard src/pic/*)

######################################################################
# creates the target and displays warnings
all: $(TARGET) warnings

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
$(TARGET): $(SOURCES) $(BIB) $(PICTURES)
	@mkdir -p ${GENDIR}
	@echo "------------initial pdf-latex run"
	@pdflatex -output-directory=${GENDIR} ${MAINTEX}
	@echo "------------Bibtex"
	@bibtex ${GENDIR}$(MAINTEX:.tex=) 
	@echo "------------Make index"
	@if test -e $(TARGET:.pdf=.idx); then makeindex $(TARGET:.pdf=.idx); fi
	@echo "------------2nd pdf-latex run"
	@pdflatex -output-directory=${GENDIR} ${MAINTEX} 1>/dev/null
	@echo "------------3rd pdf-latex run"
	@pdflatex -output-directory=${GENDIR} ${MAINTEX} 1>/dev/null

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# delete temporary files
clean:
	@rm -rf $(GENDIR)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# collect and print warnings from log-files
warnings:
	@-echo "------------ Warnings from bib:"
	@-grep arning $(TARGET:.pdf=.blg)
	@-echo "------------ Warnings from latex:"
	@-grep arning: $(TARGET:.pdf=.log) | wc -l
	@awk '/arning:/' RS= $(TARGET:.pdf=.log)
