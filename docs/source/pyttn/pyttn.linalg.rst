pyttn.linalg
============

Dense Types
-----------
.. autoclass:: pyttn.linalg.Vector
   :members:
   :special-members: __init__,__new__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyttn.linalg.Matrix
   :members:
   :special-members: __init__,__new__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyttn.linalg.Tensor3
   :members:
   :special-members: __init__,__new__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyttn.linalg.Tensor4
   :members:
   :special-members: __init__,__new__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyttn.linalg.Tensor
   :members:
   :special-members: __init__,__new__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance:

Sparse Types
------------

.. autoclass:: pyttn.linalg.SparseMatrix
   :members:
   :special-members: __init__,__new__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyttn.linalg.DiagonalMatrix
   :members:
   :special-members: __init__,__new__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyttn.linalg.CSRMatrix
   :members:
   :special-members: __init__,__new__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. autoclass:: pyttn.linalg.OrthogonalVector
   :members:
   :special-members: __init__,__new__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyttn.linalg.RandomEngine
   :members:
   :special-members: __init__,__new__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance:


Helper Functions
----------------
.. autofunction:: pyttn.linalg.available_backends

Aliases
-------

.. autoclass:: pyttn.linalg.vector
.. autoclass:: pyttn.linalg.matrix
.. autoclass:: pyttn.linalg.tensor_3
.. autoclass:: pyttn.linalg.tensor_4
.. autoclass:: pyttn.linalg.tensor
.. autoclass:: pyttn.linalg.diagonal_matrix
.. autoclass:: pyttn.linalg.csr_matrix
.. autoclass:: pyttn.linalg.orthogonal_vector
.. autoclass:: pyttn.linalg.random_engine