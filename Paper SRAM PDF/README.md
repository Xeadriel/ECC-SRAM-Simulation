# Official IEEE LaTeX Template

A clean, organized template repository for IEEE conference papers based on the official IEEE LaTeX template.

## Contents

This repository contains:
- **Main template files** ready for immediate use
- **Organized supporting files** (styles, bibliography resources, documentation)
- **Build system** with Makefile for easy compilation
- **Two bibliography approaches** (manual and BibTeX)

## Quick Start

This is a GitHub template repository and can be used as such.
So the easiest way is to:

1. Click on 'use this template' -> 'create new repository'.

2. **Edit your paper**:
   - Modify `main.tex` with your content
   - Add figures to the `img/` directory
   - Update references in `references.bib` (if you want to use the traditional approach, switch to the corresponding branch)

3. **Build your paper**:
   ```bash
   make pdf        # Full build with bibliography
   make quick      # Quick build (single pass)
   make view       # Build and view PDF (view support only for Linux and macOS)
   make clean      # Clean build artifacts
   ```

## Repository Structure

```
ieee-latex-template/
├── main.tex                   # Main template file (edit this!)
├── IEEEtran.cls               # IEEE class file
├── references.bib             # Bibliography file (BibTeX/main branch)
├── Makefile                   # Build system
├──
├── img/                       # Graphics directory
│   └── fig1.png               # Example figure
├──
├── styles/                    # BibTeX style files
│   ├── IEEEtran.bst           # Standard IEEE style
│   └── IEEEtranS.bst          # Sorted IEEE style
├──
├── bibliography/              # Bibliography resources
│   ├── IEEEabrv.bib           # Abbreviated journal names
│   ├── IEEEfull.bib           # Full journal names
│   └── IEEEexample.bib        # Example bibliography
├──
├── docs/                      # Documentation
│   ├── IEEEtran_HOWTO.pdf     # Class documentation
│   ├── IEEEtran_bst_HOWTO.pdf # BibTeX style documentation
│   └── README                 # Original README
└──
└── output/                    # Build artifacts (auto-generated)
```

## Branches

This repository has three branches for different bibliography approaches:

### `main` (Mirror of the `bibtex` Branch)
- Uses `references.bib` for bibliography management
- Automatic formatting with IEEE styles
- Easier to manage large bibliographies
- **Recommended for most users**

### `traditional` (Manual Bibliography Branch)
- Uses manual `thebibliography` environment
- Direct control over formatting
- No external dependencies
- **IEEE's official approach**

## Bibliography Management

### BibTeX Approach (main branch)
1. Add references to `references.bib`:
   ```bibtex
   @article{key,
     author = {Author Name},
     title = {Paper Title},
     journal = {Journal Name},
     year = {2023}
   }
   ```

2. Cite in your text:
   ```latex
   This is discussed in \cite{key}.
   ```

3. Build with `make pdf`

### Manual Approach (traditional branch)
1. Add entries directly in `main.tex`:
   ```latex
   \bibitem{key} Author Name, ``Paper Title,'' Journal Name, 2023.
   ```

2. Cite in your text:
   ```latex
   This is discussed in \cite{key}.
   ```

## Build System

The included Makefile provides several useful targets:

| Command | Description |
|---------|-------------|
| `make` or `make pdf` | Full build with bibliography processing |
| `make quick` | Quick build (single pass, no bibliography) |
| `make view` | Build and view PDF (auto-detects supported operating systems) |
| `make clean` | Remove build artifacts |
| `make help` | Show all available commands |

## Usage Tips

### Adding Figures
1. Place images in the `img/` directory
2. Reference them in your text:
   ```latex
   \begin{figure}[htbp]
   \centerline{\includegraphics{img/your-figure.png}}
   \caption{Your figure caption.}
   \label{fig:your-label}
   \end{figure}
   ```

### Cross-References
Use `\ref{label}` for figures, tables, and equations:
```latex
As shown in Fig.~\ref{fig:your-label}...
```

### Citations

According to IEEE standards:

- Please number citations consecutively within brackets [2]
- So only simple citations using `\cite{}`
- The sentence punctuation follows the bracket [3].

This and more information can be found in the template in the 'References' section.

## Documentation

The following documentation files can be found in the `docs` directory:

- **IEEEtran_HOWTO.pdf**: Complete guide to the IEEEtran class
- **IEEEtran_bst_HOWTO.pdf**: BibTeX style documentation
- **Original README**: Additional information about IEEE styles

## License

This template is based on the official IEEE LaTeX template. The original IEEE files are distributed under the LaTeX Project Public License (LPPL) version 1.3.

I do not own the template itself, I am just packaging it for easy of use. For more information about the license read `docs/README` in this repository.

---

**Note**: This template contains guidance text that should be removed before submission. The red text at the end of the template reminds you of this requirement.
