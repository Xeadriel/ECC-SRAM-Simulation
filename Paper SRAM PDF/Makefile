# IEEE LaTeX Template Makefile
# This Makefile helps build the IEEE conference paper template

# Main document name (without .tex extension)
MAIN = main

# Directories
OUTPUT_DIR = output
IMG_DIR = img
STYLES_DIR = styles
BIBLIOGRAPHY_DIR = bibliography

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compiler flags
LATEX_FLAGS = -interaction=nonstopmode -output-directory=$(OUTPUT_DIR)

# Default target
all: pdf

# Build PDF
pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex IEEEtran.cls
	@echo "Building $(MAIN).pdf..."
	@mkdir -p $(OUTPUT_DIR)
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@if [ -f $(OUTPUT_DIR)/$(MAIN).aux ]; then \
		$(BIBTEX) $(OUTPUT_DIR)/$(MAIN); \
		$(LATEX) $(LATEX_FLAGS) $(MAIN).tex; \
		$(LATEX) $(LATEX_FLAGS) $(MAIN).tex; \
	fi
	@echo "Build complete: $(OUTPUT_DIR)/$(MAIN).pdf"

# Quick build (single pass, no bibliography)
quick: $(MAIN).tex IEEEtran.cls
	@echo "Quick build of $(MAIN).pdf..."
	@mkdir -p $(OUTPUT_DIR)
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "Quick build complete: $(OUTPUT_DIR)/$(MAIN).pdf"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(OUTPUT_DIR)
	@echo "Clean complete"

# View the PDF (auto-detect macOS/Linux)
view: pdf
	@if [ -f $(OUTPUT_DIR)/$(MAIN).pdf ]; then \
		if [ "$$(uname)" = "Darwin" ]; then \
			open $(OUTPUT_DIR)/$(MAIN).pdf; \
		elif [ "$$(uname)" = "Linux" ]; then \
			xdg-open $(OUTPUT_DIR)/$(MAIN).pdf; \
		else \
			echo "Unsupported OS. Please open $(OUTPUT_DIR)/$(MAIN).pdf manually."; \
		fi; \
	else \
		echo "PDF not found. Run 'make pdf' first."; \
	fi

# Help
help:
	@echo "Available targets:"
	@echo "  pdf        - Build the complete PDF (default)"
	@echo "  quick      - Quick build (single pass, no bibliography)"
	@echo "  clean      - Remove build artifacts"
	@echo "  view       - Build and view PDF (auto-detect macOS/Linux)"
	@echo "  help       - Show this help message"

# Phony targets
.PHONY: all pdf quick clean view help
