#build the rst file from the pyttn source files
sphinx-apidoc -o docs/source/pyttn pyttn 
sphinx-apidoc -o docs/source/ttnpp python 

# sphinx-apidoc doesn't allow setting maxdepth on subpackages
sed -i "s/:maxdepth:.*/:maxdepth: 2/g" docs/source/pyttn/*.rst
sed -i "s/:maxdepth:.*/:maxdepth: 2/g" docs/source/ttnpp/*.rst

sed -i "/:members:/a \ \ \ \ :special-members:\ __call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__" docs/source/pyttn/*.rst
sed -i "/:special-members:/a \ \ \ \ :exclude-members:\ __init__,__module__,__annotations__" docs/source/pyttn/*.rst

# sphinx toctree is very indentation sensitive, make it uniform
sed -i "s/    /   /g" docs/source/pyttn/*.rst
sed -i "s/    /   /g" docs/source/ttnpp/*.rst


