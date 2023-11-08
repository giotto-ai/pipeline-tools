#    Copyright (C) 2023  Bruno Da Rocha Carvalho, Gabriel Catel Torres Arzur
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from ..function_parser import _parse_func

class GetAttr:
    """Handle all the getattr.
    
    A getattr cannot be a layer because the result of a getattr is not a tensor.
    So we have to attach the getattr with the Layer who need to use a getattr on his attribute.
    For that we create an object who save all the information, parent, child and attribute for the getattr.
    :param node: The actual getattr node
    :type node: torch.fx.node.Node
    :param trace: The trace of the model generated by torch fx
    :type trace: torch.fx.graph._node_list
    """

    def __init__(self, node, trace, child = None):
        """Constructor."""
        self.getitem_idx = None
        self.parent = node.args[0]
        self.attr = node.args[1]
        self.stash_needed = False
        self.node = node
        self.position = 0
        
        self.child = child
        if self.child is None:
            for _node in trace:
                if node in _node.args:
                    if str(_node).find("getitem") >= 0:
                        node = _node
                    else:
                        self.child = _node
                        self.position = _node.args.index(node)
                        break
        # Verifiy if the result of the getattr need to be stashed.
        prev_node = None
        for _node in trace:
            if _node == self.child:
                if prev_node != self.parent:
                    self.stash_needed = True
                    self.getattr_string = f"{self.parent}.{self.attr}"
                else:
                    self.getattr_string = f"input.{self.attr}"

            if not str(_node).find("getitem") >= 0 and self.node != _node:
                prev_node = _node
    
    def get_position(self) -> int:
        """Return the position of the attribute in the list of param during the call
        
        :return: Return the position of the attribute in the list of param during the call
        :rtype: int
        """
        return self.position

    def get_attr_name(self) -> str:
        """Return the name of the attribute used.

        :return: Return the name of the attribute used
        :rtype: str
        """
        return self.getattr_string

    def get_child(self) -> torch.fx.node.Node:
        """Return the node in which the getattr will be done.

        :return: Return the child node of the getattr
        :rtype: torch.fx.node.Node
        """
        return self.child

    def get_parent(self) -> torch.fx.node.Node:
        """Return the node of the ret value on which will be done the getattr.

        :return: Return the parent node of the getattr
        :rtype: torch.fx.node.Node
        """
        if self.stash_needed:
            return self.parent
        else:
            return 'input'

    def add_getitem(self, idx):
        """Allow to add a getitem on a getattr call.
        
        For example : input.shape[0]

        :param idx: The index of the getitem
        :type idx: int
        """
        self.getitem_idx = idx
        self.getattr_string += f"[{self.getitem_idx}]"

    def is_stash_needed(self) -> bool:
        """Return True if the getattr need to be stashed, else False.
        
        :return: If stash is needed
        :rtype: bool
        """
        return self.stash_needed

    def __str__(self) -> str:
        """Allow to print easily all the information of a Getattr.

        :return: String to print
        :rtype: str
        """
        print_str = "GetAttr info : \n"
        print_str += f"    Attribute {self.attr} of ret of node {self.parent}\n"
        print_str += f"    This getattr is needed in {self.child}\n"
        print_str += f"    If not empty use those specific idx of the result {self.getitem_idx}\n"
        if self.stash_needed:
            print_str += f"    A new stash have to be add on {self.parent} for {self.child}\n"
        else:
            print_str += f"    No new stash need to be added on {self.parent}, the dest is directly connected\n"

        return print_str