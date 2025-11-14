Topology
========

.. autoclass:: pyttn.ttns.ttns.ntree
   :members:
   :special-members: __new__,__init__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance:
   
.. autoclass:: pyttn.ttns.ttns.ntreeNode
   :members:
   :special-members: __new__,__init__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance:

Generating Specific Tree Topologies
-----------------------------------

.. autoclass:: pyttn.ttns.ttns.ntreeBuilder
   :members:
   :special-members: __new__,__init__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance:

Automated Tree Generation
-------------------------

.. autofunction:: pyttn.ttns.topology.generate_spanning_tree

.. autofunction:: pyttn.ttns.topology.generate_hierarchical_clustering_tree

.. autofunction:: pyttn.ttns.topology.convert_nx_to_subtree

.. autofunction:: pyttn.ttns.topology.convert_nx_to_tree


Updating Bond Dimension
-----------------------

.. autofunction:: pyttn.ttns.topology.set_bond_dimensions

.. autofunction:: pyttn.ttns.topology.set_dims

.. autofunction:: pyttn.ttns.topology.set_topology_properties

.. autoclass:: pyttn.ttns.topology.NodeSumSetter
   :members:
   :special-members: __new__,__init__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyttn.ttns.topology.NodeIncrementSetter
   :members:
   :special-members: __new__,__init__,__call__,__copy__,__deepcopy__,__radd__,__rsub__,__rdiv__,__rmul__,__add__,__sub__,__div__,__mul__,__iadd__,__isub__,__idiv__,__imul__, __iter__,__len__,__setitem__,__getitem__,__str__
   :exclude-members: __module__,__annotations__
   :undoc-members:
   :show-inheritance: