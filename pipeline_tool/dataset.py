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

class PipelineDataset(torch.utils.data.Dataset):
    def __init__(self, size, input_shape, output_shape, dtype="long"):
        """Initialize a PipelineDataset object.
        
        :param size: Size of the dataset.
        :type size: int
        :param input_shape: Shape of the input data.
        :type input_shape: tuple
        :param output_shape: Shape of the output data.
        :type output_shape: tuple
        """
        self.size = size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dtype = self.parse_dtype(dtype)

    def parse_dtype(self, dtype):
        """Map data type in string format to return data type of API torch

        :param dtype: Data type
        :type dtype: str
        :return: Data type of API torch
        :rtype: dtype
        """
        data_type_mapping = {
            'float32': torch.float32,
            'float': torch.float32,
            'float64': torch.float64,
            'double': torch.float64,
            'float16': torch.float16,
            'half': torch.float16,
            'bfloat16': torch.bfloat16,
            'complex32': torch.complex32,
            'chalf': torch.complex32,
            'complex64': torch.complex64,
            'cfloat': torch.complex64,
            'complex128': torch.complex128,
            'cdouble': torch.complex128,
            'uint8': torch.uint8,
            'byte': torch.uint8,
            'int8': torch.int8,
            'char': torch.int8,
            'int16': torch.int16,
            'short': torch.int16,
            'int32': torch.int32,
            'int': torch.int32,
            'int64': torch.int64,
            'long': torch.int64,
            'bool': torch.bool,
            'quint8': torch.quint8,
            'qint8': torch.qint8,
            'qint32': torch.qint32,
            'quint4x2': torch.quint4x2
        }
        data_type_str = dtype.lower()  # Convertir en minuscules pour être insensible à la casse
        return data_type_mapping.get(dtype, None)

    def __len__(self):
        """Return the length of the dataset.
        
        :return: Length of the dataset.
        :rtype: int
        """
        return self.size

    def __getitem__(self, idx):
        """Get an item from the dataset at the given index.
        
        :param idx: Index of the item.
        :type idx: int
        :return: Tuple of input data and target data.
        :rtype: tuple
        """
        data = torch.randn(*self.input_shape)
        target = torch.randint(0, 2, self.output_shape, dtype=self.dtype)
        return data, target

