#build the rst file from the pyttn source files
sphinx-apidoc -o docs/source pyttn -f


# sphinx-apidoc doesn't allow setting maxdepth on subpackages
sed -i "s/:maxdepth:.*/:maxdepth: 2/g" docs/source/*.rst
# sphinx toctree is very indentation sensitive, make it uniform
sed -i "s/    /   /g" docs/source/*.rst


