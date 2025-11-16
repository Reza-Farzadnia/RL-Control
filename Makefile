SHELL := /bin/bash

.PHONY: docs clean

docs:
	./scripts/build_docs.sh

clean:
	@echo "Cleaning LaTeX aux and built PDFs..."
	rm -rf tex/build docs/models/*.pdf

