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

from .LayerClass import Layer
from ..function_parser import _parse_func


class CallFunction(Layer):
    """Handle all the traced layer define as "call_function".

    A call function is a layer with no init, he just executes a function or method on tensors.
    For example a torch.add or a call to the method expand.

    :param node: Is the actual traced node from torch.fx.
    :type node: torch.fx.node.Node
    :param trace: Is the complete trace of torch.fx of the model.
    :type trace: torch.fx.graph._node_list
    :param prev_node: Is the just previous node in the trace before the actual traced node.
    :type node: torch.fx.node.Node
    """
    
    def __init__(self, node, trace, prev_node):
        """Constructor."""
        super().__init__(node, trace, prev_node)
        # This call will return the exact function or method call
        self.function_call = _parse_func(node, self.args)

    def update_arg_by_attr(self, new_arg, position):
        """Allow to update argument with a specific attribute.

        For example t.shape. We need to reset the function call as we change the args.

        :param old_arg: Old argument to replace.
        :type old_arg: torch.fx.node.Node
        :param new_arg: New arg containing the attribute. (For ex .shape)
        :type new_arg: str
        """
        super().update_arg_by_attr(new_arg, position)
        self.function_call = _parse_func(self.node, self.args)

    def get_declaration(self) -> str:
        """Generate and return the full class generate for a layer containing a call method or call function.
        
        For example :
        .. code-block:: python
            @skippable...
            class {self.name}_layer(nn.Module):
                def forward(self, input):
                    ... = yield pop...
                    ret = call function or call method (input.expand(16, -1, -1))
                    yield stash ...
                    return ret

        :return: The full declaration of a Layer containing a CallFunction
        :rtype: str
        """
        string = self.generate_class()
        string += self.generate_forward(self.function_call)

        return string

    def __str__(self) -> str:
        """Allow to print easily all the information of a layer.
        
        It adds a print of the function executed at the overload of the Layer class.

        :return: String to print
        :rtype: str
        """
        print_str = super().__str__()
        print_str += f"    The function call is {self.function_call}\n\n"
        return print_str
