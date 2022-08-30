from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple, Union
from graphviz import Digraph, nohtml

try:  # pragma: no cover
    from graphviz.exceptions import ExecutableNotFound
except ImportError:  # pragma: no cover
    # noinspection PyProtectedMember
    from graphviz import ExecutableNotFound
from pkg_resources import get_distribution
from subprocess import SubprocessError

_NODE_VAL_TYPES = Any
NodeValue = Any  # Union[float, int, str]
NodeValueList = Union[
    List[Optional[float]],
    List[Optional[int]],
    List[Optional[str]],
    List[float],
    List[int],
    List[str],
]
_SVG_XML_TEMPLATE = """
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
<style>
    .value {{
        font: 300 16px sans-serif;
        text-align: center;
        dominant-baseline: middle;
        text-anchor: middle;
    }}
    .node {{
        fill: lightgray;
        stroke-width: 1;
    }}
</style>
<g stroke="#000000">
{body}
</g>
</svg>
"""

# Create a Custom Binary Tree
class Node:
    def __init__(
        self,
        value: NodeValue,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
    ) -> None:
        self.value = self.val = value
        self.left = left
        self.right = right
    
    def __repr__(self) -> str:
        return "State({}), Action({}), Belief({})".format(self.val.get_state(), self.val.get_action(), self.val.get_belief())

    def __str__(self) -> str:
        lines = _build_tree_string(self, 0, False, "-")[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    def __iter__(self) -> Iterator["Node"]:
        """Iterate through the nodes in the binary tree in level-order_.

        .. _level-order:
            https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search

        :return: Node iterator.
        :rtype: Iterator[binarytree.Node]

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.left = Node(4)
            >>> root.left.right = Node(5)
            >>>
            >>> print(root)
            <BLANKLINE>
                __1
               /   \\
              2     3
             / \\
            4   5
            <BLANKLINE>
            >>> list(root)
            [Node(1), Node(2), Node(3), Node(4), Node(5)]
        """
        current_nodes = [self]

        while len(current_nodes) > 0:
            next_nodes = []

            for node in current_nodes:
                yield node

                if node.left is not None:
                    next_nodes.append(node.left)
                if node.right is not None:
                    next_nodes.append(node.right)

            current_nodes = next_nodes

    def __len__(self) -> int:
        """Return the total number of nodes in the binary tree.

        :return: Total number of nodes.
        :rtype: int

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>>
            >>> len(root)
            3

        .. note::
            This method is equivalent to :attr:`binarytree.Node.size`.
        """
        return sum(1 for _ in iter(self))

    def __getitem__(self, index: int) -> "Node":
        """Return the node (or subtree) at the given level-order_ index.

        .. _level-order:
            https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search

        :param index: Level-order index of the node.
        :type index: int
        :return: Node (or subtree) at the given index.
        :rtype: binarytree.Node
        :raise binarytree.exceptions.NodeIndexError: If node index is invalid.
        :raise binarytree.exceptions.NodeNotFoundError: If the node is missing.

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)       # index: 0, value: 1
            >>> root.left = Node(2)  # index: 1, value: 2
            >>> root.right = Node(3) # index: 2, value: 3
            >>>
            >>> root[0]
            Node(1)
            >>> root[1]
            Node(2)
            >>> root[2]
            Node(3)
            >>> root[3]  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
             ...
            binarytree.exceptions.NodeNotFoundError: node missing at index 3
        """
        if not isinstance(index, int) or index < 0:
            raise NodeIndexError("node index must be a non-negative int")

        current_nodes: List[Optional[Node]] = [self]
        current_index = 0
        has_more_nodes = True

        while has_more_nodes:
            has_more_nodes = False
            next_nodes: List[Optional[Node]] = []

            for node in current_nodes:
                if current_index == index:
                    if node is None:
                        break
                    else:
                        return node
                current_index += 1

                if node is None:
                    next_nodes.append(None)
                    next_nodes.append(None)
                    continue
                next_nodes.append(node.left)
                next_nodes.append(node.right)
                if node.left is not None or node.right is not None:
                    has_more_nodes = True

            current_nodes = next_nodes

        raise NodeNotFoundError("node missing at index {}".format(index))

    def __setitem__(self, index: int, node: "Node") -> None:
        """Insert a node (or subtree) at the given level-order_ index.

        * An exception is raised if the parent node is missing.
        * Any existing node or subtree is overwritten.
        * Root node (current node) cannot be replaced.

        .. _level-order:
            https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search

        :param index: Level-order index of the node.
        :type index: int
        :param node: Node to insert.
        :type node: binarytree.Node
        :raise binarytree.exceptions.NodeTypeError: If new node is not an
            instance of :class:`binarytree.Node`.
        :raise binarytree.exceptions.NodeNotFoundError: If parent is missing.
        :raise binarytree.exceptions.NodeModifyError: If user attempts to
            overwrite the root node (current node).

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)       # index: 0, value: 1
            >>> root.left = Node(2)  # index: 1, value: 2
            >>> root.right = Node(3) # index: 2, value: 3
            >>>
            >>> root[0] = Node(4)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
             ...
            binarytree.exceptions.NodeModifyError: cannot modify the root node

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)       # index: 0, value: 1
            >>> root.left = Node(2)  # index: 1, value: 2
            >>> root.right = Node(3) # index: 2, value: 3
            >>>
            >>> root[11] = Node(4)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
             ...
            binarytree.exceptions.NodeNotFoundError: parent node missing at index 5

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)       # index: 0, value: 1
            >>> root.left = Node(2)  # index: 1, value: 2
            >>> root.right = Node(3) # index: 2, value: 3
            >>>
            >>> root[1] = Node(4)
            >>>
            >>> root.left
            Node(4)
        """
        if index == 0:
            raise NodeModifyError("cannot modify the root node")

        parent_index = (index - 1) // 2
        try:
            parent = self.__getitem__(parent_index)
        except NodeNotFoundError:
            raise NodeNotFoundError(
                "parent node missing at index {}".format(parent_index)
            )

        setattr(parent, _ATTR_LEFT if index % 2 else _ATTR_RIGHT, node)

    def __delitem__(self, index: int) -> None:
        """Remove the node (or subtree) at the given level-order_ index.

        * An exception is raised if the target node is missing.
        * The descendants of the target node (if any) are also removed.
        * Root node (current node) cannot be deleted.

        .. _level-order:
            https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search

        :param index: Level-order index of the node.
        :type index: int
        :raise binarytree.exceptions.NodeNotFoundError: If the target node or
            its parent is missing.
        :raise binarytree.exceptions.NodeModifyError: If user attempts to
            delete the root node (current node).

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)          # index: 0, value: 1
            >>> root.left = Node(2)     # index: 1, value: 2
            >>> root.right = Node(3)    # index: 2, value: 3
            >>>
            >>> del root[0]  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
             ...
            binarytree.exceptions.NodeModifyError: cannot delete the root node

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)          # index: 0, value: 1
            >>> root.left = Node(2)     # index: 1, value: 2
            >>> root.right = Node(3)    # index: 2, value: 3
            >>>
            >>> del root[2]
            >>>
            >>> root[2]  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
             ...
            binarytree.exceptions.NodeNotFoundError: node missing at index 2
        """
        if index == 0:
            raise NodeModifyError("cannot delete the root node")

        parent_index = (index - 1) // 2
        try:
            parent = self.__getitem__(parent_index)
        except NodeNotFoundError:
            raise NodeNotFoundError("no node to delete at index {}".format(index))

        child_attr = _ATTR_LEFT if index % 2 == 1 else _ATTR_RIGHT
        if getattr(parent, child_attr) is None:
            raise NodeNotFoundError("no node to delete at index {}".format(index))

        setattr(parent, child_attr, None)

    def _repr_svg_(self) -> str:  # pragma: no cover
        """Display the binary tree using Graphviz (used for `Jupyter notebooks`_).

        .. _Jupyter notebooks: https://jupyter.org
        """
        try:
            try:
                # noinspection PyProtectedMember
                return str(self.graphviz()._repr_svg_())
            except AttributeError:
                # noinspection PyProtectedMember
                return str(self.graphviz()._repr_image_svg_xml())

        except (SubprocessError, ExecutableNotFound, FileNotFoundError):
            return self.svg()

    def svg(self, node_radius: int = 16) -> str:
        """Generate SVG XML.

        :param node_radius: Node radius in pixels (default: 16).
        :type node_radius: int
        :return: Raw SVG XML.
        :rtype: str
        """
        tree_height = self.height
        scale = node_radius * 3
        xml: Deque[str] = deque()

        def scale_x(x: int, y: int) -> float:
            diff = tree_height - y
            x = 2 ** (diff + 1) * x + 2**diff - 1
            return 1 + node_radius + scale * x / 2

        def scale_y(y: int) -> float:
            return scale * (1 + y)

        def add_edge(parent_x: int, parent_y: int, node_x: int, node_y: int) -> None:
            xml.appendleft(
                '<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'.format(
                    x1=scale_x(parent_x, parent_y),
                    y1=scale_y(parent_y),
                    x2=scale_x(node_x, node_y),
                    y2=scale_y(node_y),
                )
            )

        def add_node(node_x: int, node_y: int, node_value: NodeValue) -> None:
            x, y = scale_x(node_x, node_y), scale_y(node_y)
            xml.append(f'<circle class="node" cx="{x}" cy="{y}" r="{node_radius}"/>')
            xml.append(f'<text class="value" x="{x}" y="{y}">{node_value}</text>')

        current_nodes = [self.left, self.right]
        has_more_nodes = True
        y = 1

        add_node(0, 0, self.value)

        while has_more_nodes:

            has_more_nodes = False
            next_nodes: List[Optional[Node]] = []

            for x, node in enumerate(current_nodes):
                if node is None:
                    next_nodes.append(None)
                    next_nodes.append(None)
                else:
                    if node.left is not None or node.right is not None:
                        has_more_nodes = True

                    add_edge(x // 2, y - 1, x, y)
                    add_node(x, y, node.value)

                    next_nodes.append(node.left)
                    next_nodes.append(node.right)

            current_nodes = next_nodes
            y += 1

        return _SVG_XML_TEMPLATE.format(
            width=scale * (2**tree_height),
            height=scale * (2 + tree_height),
            body="\n".join(xml),
        )

    def graphviz(self, *args: Any, **kwargs: Any) -> Digraph:  # pragma: no cover
        """Return a graphviz.Digraph_ object representing the binary tree.

        This method's positional and keyword arguments are passed directly into the
        Digraph's **__init__** method.

        :return: graphviz.Digraph_ object representing the binary tree.
        :raise binarytree.exceptions.GraphvizImportError: If graphviz is not installed

        .. code-block:: python

            >>> from binarytree import tree
            >>>
            >>> t = tree()
            >>>
            >>> graph = t.graphviz()    # Generate a graphviz object
            >>> graph.body              # Get the DOT body
            >>> graph.render()          # Render the graph

        .. _graphviz.Digraph: https://graphviz.readthedocs.io/en/stable/api.html#digraph
        """
        if "node_attr" not in kwargs:
            kwargs["node_attr"] = {
                "shape": "record",
                "style": "filled, rounded",
                "color": "lightgray",
                "fillcolor": "lightgray",
                "fontcolor": "black",
            }
        digraph = Digraph(*args, **kwargs)

        for node in self:
            node_id = str(id(node))

            digraph.node(node_id, nohtml(f"<l>|<v> {node.value}|<r>"))

            if node.left is not None:
                digraph.edge(f"{node_id}:l", f"{id(node.left)}:v")

            if node.right is not None:
                digraph.edge(f"{node_id}:r", f"{id(node.right)}:v")

        return digraph

    def pprint(self, index: bool = False, delimiter: str = "-") -> None:
        """Pretty-print the binary tree.

        :param index: If set to True (default: False), display level-order_
            indexes using the format: ``{index}{delimiter}{value}``.
        :type index: bool
        :param delimiter: Delimiter character between the node index and
            the node value (default: '-').
        :type delimiter: str

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)              # index: 0, value: 1
            >>> root.left = Node(2)         # index: 1, value: 2
            >>> root.right = Node(3)        # index: 2, value: 3
            >>> root.left.right = Node(4)   # index: 4, value: 4
            >>>
            >>> root.pprint()
            <BLANKLINE>
              __1
             /   \\
            2     3
             \\
              4
            <BLANKLINE>
            >>> root.pprint(index=True)     # Format: {index}-{value}
            <BLANKLINE>
               _____0-1_
              /         \\
            1-2_        2-3
                \\
                4-4
            <BLANKLINE>

        .. note::
            If you do not need level-order_ indexes in the output string, use
            :func:`binarytree.Node.__str__` instead.

        .. _level-order:
            https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
        """
        lines = _build_tree_string(self, 0, index, delimiter)[0]
        print("\n" + "\n".join((line.rstrip() for line in lines)))

    # def validate(self) -> None:
    #     """Check if the binary tree is malformed.

    #     :raise binarytree.exceptions.NodeReferenceError: If there is a
    #         cyclic reference to a node in the binary tree.
    #     :raise binarytree.exceptions.NodeTypeError: If a node is not an
    #         instance of :class:`binarytree.Node`.
    #     :raise binarytree.exceptions.NodeValueError: If node value is invalid.

    #     **Example**:

    #     .. doctest::

    #         >>> from binarytree import Node
    #         >>>
    #         >>> root = Node(1)
    #         >>> root.left = Node(2)
    #         >>> root.right = root  # Cyclic reference to root
    #         >>>
    #         >>> root.validate()  # doctest: +IGNORE_EXCEPTION_DETAIL
    #         Traceback (most recent call last):
    #          ...
    #         binarytree.exceptions.NodeReferenceError: cyclic node reference at index 0
    #     """
    #     has_more_nodes = True
    #     nodes_seen = set()
    #     current_nodes: List[Optional[Node]] = [self]
    #     node_index = 0  # level-order index

    #     while has_more_nodes:

    #         has_more_nodes = False
    #         next_nodes: List[Optional[Node]] = []

    #         for node in current_nodes:
    #             if node is None:
    #                 next_nodes.append(None)
    #                 next_nodes.append(None)
    #             else:
    #                 if node in nodes_seen:
    #                     raise NodeReferenceError(
    #                         f"cyclic reference at Node({node.val}) "
    #                         + f"(level-order index {node_index})"
    #                     )
    #                 if not isinstance(node, Node):
    #                     raise NodeTypeError(
    #                         "invalid node instance at index {}".format(node_index)
    #                     )
    #                 if not isinstance(node.val, _NODE_VAL_TYPES):
    #                     raise NodeValueError(
    #                         "invalid node value at index {}".format(node_index)
    #                     )
    #                 if not isinstance(node.value, _NODE_VAL_TYPES):  # pragma: no cover
    #                     raise NodeValueError(
    #                         "invalid node value at index {}".format(node_index)
    #                     )
    #                 if node.left is not None or node.right is not None:
    #                     has_more_nodes = True

    #                 nodes_seen.add(node)
    #                 next_nodes.append(node.left)
    #                 next_nodes.append(node.right)

    #             node_index += 1

    #         current_nodes = next_nodes

    def equals(self, other: "Node") -> bool:
        """Check if this binary tree is equal to other binary tree.

        :param other: Root of the other binary tree.
        :type other: binarytree.Node
        :return: True if the binary trees are equal, False otherwise.
        :rtype: bool
        """
        stack1: List[Optional[Node]] = [self]
        stack2: List[Optional[Node]] = [other]

        while stack1 or stack2:
            node1 = stack1.pop()
            node2 = stack2.pop()

            if node1 is None and node2 is None:
                continue
            elif node1 is None or node2 is None:
                return False
            elif not isinstance(node2, Node):
                return False
            else:
                if node1.val != node2.val:
                    return False
                stack1.append(node1.right)
                stack1.append(node1.left)
                stack2.append(node2.right)
                stack2.append(node2.left)

        return True

    def clone(self) -> "Node":
        """Return a clone of this binary tree.

        :return: Root of the clone.
        :rtype: binarytree.Node
        """
        other = Node(self.val)

        stack1 = [self]
        stack2 = [other]

        while stack1 or stack2:
            node1 = stack1.pop()
            node2 = stack2.pop()

            if node1.left is not None:
                node2.left = Node(node1.left.val)
                stack1.append(node1.left)
                stack2.append(node2.left)

            if node1.right is not None:
                node2.right = Node(node1.right.val)
                stack1.append(node1.right)
                stack2.append(node2.right)

        return other

    @property
    def values(self) -> List[Optional[NodeValue]]:
        """Return the `list representation`_ of the binary tree.

        .. _list representation:
            https://en.wikipedia.org/wiki/Binary_tree#Arrays

        :return: List representation of the binary tree, which is a list of node values
            in breadth-first order starting from the root. If a node is at index i, its
            left child is always at 2i + 1, right child at 2i + 2, and parent at index
            floor((i - 1) / 2). None indicates absence of a node at that index. See
            example below for an illustration.
        :rtype: [float | int | None]

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.left.left = Node(3)
            >>> root.left.left.left = Node(4)
            >>> root.left.left.right = Node(5)
            >>>
            >>> root.values
            [1, 2, None, 3, None, None, None, 4, 5]
        """
        current_nodes: List[Optional[Node]] = [self]
        has_more_nodes = True
        node_values: List[Optional[NodeValue]] = []

        while has_more_nodes:
            has_more_nodes = False
            next_nodes: List[Optional[Node]] = []

            for node in current_nodes:
                if node is None:
                    node_values.append(None)
                    next_nodes.append(None)
                    next_nodes.append(None)
                else:
                    if node.left is not None or node.right is not None:
                        has_more_nodes = True

                    node_values.append(node.val)
                    next_nodes.append(node.left)
                    next_nodes.append(node.right)

            current_nodes = next_nodes

        # Get rid of trailing None values
        while node_values and node_values[-1] is None:
            node_values.pop()

        return node_values

def _build_tree_string(
    root: Optional[Node],
    curr_index: int,
    include_index: bool = False,
    delimiter: str = "-",
    ) -> Tuple[List[str], int, int, int]:

    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if include_index:
        node_repr = "{}{}{}{}".format(curr_index, delimiter, root.val.get_state()[0],
                                     root.val.get_action)
    else:
        node_repr = f'{str(round(root.val.get_state()[0], 2))}, {str(root.val.get_action())}, {str(root.val.get_belief())}'

    new_root_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = _build_tree_string(
        root.left, 2 * curr_index + 1, include_index, delimiter)
    r_box, r_box_width, r_root_start, r_root_end = _build_tree_string(
        root.right, 2 * curr_index + 2, include_index, delimiter)

    # Draw the branch connecting the current root node to the left sub-box
    # Pad the line with whitespaces where necessary
    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(" " * (l_root + 1))
        line1.append("_" * (l_box_width - l_root))   ##
        line2.append(" " * l_root + "/")
        line2.append(" " * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    # Draw the representation of the current root node
    line1.append(node_repr)
    line2.append(" " * new_root_width)

    # Draw the branch connecting the current root node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append("_" * r_root)
        line1.append(" " * (r_box_width - r_root + 1))
        line2.append(" " * r_root + "\\")
        line2.append(" " * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = " " * gap_size
    new_box = ["".join(line1), "".join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else " " * l_box_width
        r_line = r_box[i] if i < len(r_box) else " " * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its root repr positions
    return new_box, len(new_box[0]), new_root_start, new_root_end