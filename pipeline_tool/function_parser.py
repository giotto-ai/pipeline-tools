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

def _generate_function(var_names, func) -> str:
    """Return a string containg a call to a torch function.
    
    :param var_names: All the var that will be passed to the torch function
    :type var_names: list
    :param func: Name of the torch function to call
    :type func: str
    """
    string = f"torch.{func}("
    for var in var_names:        
        if isinstance(var, tuple):
            string += f"{var[0]}={var[1]}, "
        
        else:
            if isinstance(var, list):
                string += f"["
                for elem in var:
                    string += f"{elem}, "
                string = string[:-2]
                string += f"], "
            else:
                string += f"{var}, "
    string = string[:-2]
    string += ")"
    return string

def _generate_method(var_names, meth) -> str:
    """Return a string containg a call to a tensor method.
    
    :param var_names: All the var that will be passed to the tensor method
    :type var_names: list
    :param func: Name of the tensor method
    :type func: str
    """
    string = f"{var_names[0]}.{meth}("
    if len(var_names) > 1:
        for var in var_names[1:]:
            if isinstance(var, tuple):
                string += f"{var[0]}={var[1]}, "
            else:
                if isinstance(var, list):
                    string += f"["
                    for elem in var:
                        string += f"{elem}, "
                    string = string[:-2]
                    string += f"], "
                else:
                    string += f"{var}, "
        string = string[:-2]
    string += ")"
    return string

def gen_var_names(var_names, scheme, has_kwargs) -> list: 
    """Return the modify arg list with the specified scheme.

    :param var_names: List of argument.
    :type var_names: list of list, tuple, torch.fx.Node.node, int and string
    :param scheme: Description of argument repartition by group.
    :type scheme: list of int
    :param has_kwargs: Number of kwargs in var_names
    :type has_kwargs: int
    :return: List of grouped var_names with the description of scheme
    :rtype: list of list, tuple, torch.fx.Node.node, int and string
    """
    arg_cnt = 0
    new_var_names = []
    for n in scheme:
        if n > 1:
            new_var_names.append(var_names[arg_cnt:arg_cnt+n])
        else:
            new_var_names.append(var_names[arg_cnt])
        arg_cnt += n
    if has_kwargs:
        new_var_names.extend(var_names[-has_kwargs:])

    return new_var_names

def _parse_func(key, var_names) -> str:
    """Return a string containg the full call to a method or function for torch tensor.
    
    :param key: Node that have his op call_method or call_function
    :type key: torch.fx.Node.node
    :param var_names: All the var that will be passed to the torch function
    :type var_names: list
    """
    name = str(key)
    try:
        if name.find("add") >= 0:  
            return _generate_function(var_names, "add")

        elif name.find("sub") >= 0:  
            return _generate_function(var_names, "sub")

        elif name.find("mul") >= 0 and name.find("matmul") < 0:  
            return _generate_function(var_names, "mul")

        elif name.find("floordiv") >= 0 or name.find("floor_divide") >= 0:   
            return _generate_function(var_names, "floor_divide")

        elif name.find("truediv") >= 0 or name.find("true_divide") >= 0:  
            return _generate_function(var_names, "true_divide")

        elif name.find("cat") >= 0:
            tmp_var = var_names
            has_kwargs = 0

            while isinstance(tmp_var[-1], tuple):
                tmp_var = tmp_var[:-1]
                has_kwargs += 1

            if isinstance(tmp_var[-1], int):
                scheme = [len(tmp_var[:-1]), 1]
            else:
                scheme = [len(tmp_var)]

            var_names = gen_var_names(var_names, scheme, has_kwargs)

            return _generate_function(var_names, "cat")

        elif name.find("split") >= 0:  
            tmp_var = var_names
            has_kwargs = 0

            while isinstance(tmp_var[-1], tuple):
                tmp_var = tmp_var[:-1]
                has_kwargs += 1

            scheme = [1, len(tmp_var[1:])]
            
            var_names = gen_var_names(var_names, scheme, has_kwargs)

            return _generate_function(var_names, "split")

        elif name.find("flatten") >= 0:  
            return _generate_function(var_names, "flatten")

        elif name.find("relu") >= 0:  
            return _generate_function(var_names, "nn.functional.relu")

        elif name.find("matmul") >= 0:  
            return _generate_function(var_names, "matmul") 

        elif name.find("transpose") >= 0:  
            return _generate_function(var_names, "transpose")

        elif name.find("expand") >= 0:  
            return _generate_method(var_names, "expand")

        elif name.find("reshape") >= 0: 
            tmp_var = var_names

            has_kwargs = 0

            while isinstance(tmp_var[-1], tuple):
                tmp_var = tmp_var[:-1]
                has_kwargs += 1
            
            scheme = [1, len(tmp_var[1:])]
            
            var_names = gen_var_names(var_names, scheme, 0)

            return _generate_function(var_names, "reshape")

        elif name.find("permute") >= 0: 
            tmp_var = var_names
            has_kwargs = 0

            while isinstance(tmp_var[-1], tuple):
                tmp_var = tmp_var[:-1]
                has_kwargs += 1
            
            scheme = [1, len(tmp_var[1:])]
            
            var_names = gen_var_names(var_names, scheme, 0)

            return _generate_function(var_names, "permute")

        elif name.find("softmax") >= 0:  
            return _generate_function(var_names, "nn.functional.softmax")

        elif name.find("view") >= 0:  
            return _generate_method(var_names, "view")

        elif name.find("to") >= 0:  
            return _generate_method(var_names, "to")

        elif name.find("pow") >= 0:  
            return _generate_function(var_names, "pow")

        elif name.find("mean") >= 0:   
            return _generate_function(var_names, "mean")

        elif name.find("rsqrt") >= 0:  
            return _generate_function(var_names, "rsqrt")

        elif name.find("unsqueeze") >= 0:  
            return _generate_function(var_names, "unsqueeze")

        elif name.find("squeeze") >= 0:  
            return _generate_method(var_names, "squeeze")

        elif name.find("float") >= 0: 
            raise AssertionError(f"Function not handled: {name}")

        elif name.find("type_as") >= 0: 
            raise AssertionError(f"Function not handled: {name}")

        elif name.find("dropout") >= 0:  
            return _generate_function(var_names, "nn.functional.dropout")

        elif name.find("contiguous") >= 0: 
            raise AssertionError(f"Function not handled: {name}")

        elif name.find("tanh") >= 0:  
            return _generate_function(var_names, "nn.functional.tanh")

        elif name.find("gelu") >= 0:  
            return _generate_function(var_names, "nn.functional.gelu")

        elif name.find("size") >= 0:  
            return _generate_method(var_names, "size")

        else:
            raise AssertionError(f"Unknown function or method: {name}")
        
    except AssertionError as e:
        print(f"Error : {e}") 
        exit()