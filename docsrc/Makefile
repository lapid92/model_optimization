# Minimal makefile for Sphinx documentation
#

github:
	@rm -rf ../docs/
	@mkdir ../docs
	@make html
	@cp -a ../docs/html/. ../docs
	@rm -rf ../docs/html/
	@rm -rf ../docs/doctrees/
	@find ../docs -type f -exec sed -i 's|_images/|images/|g' {} +
	@find ../docs -type f -exec sed -i 's|_static/|static/|g' {} +
	@rm -rf ../docs/images
	@rm -rf ../docs/static
	@mv ../docs/_images/ ../docs/images
	@mv ../docs/_static/ ../docs/static



# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = ../docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
