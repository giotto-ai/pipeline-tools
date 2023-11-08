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

class PipelineConfig:
    """Create a configuration object for pipeline tool.

    :param input_shape: Input shape of the model
    :type input_shape: list of int
    :param output_shape: Output shape of the model
    :type output_shape: list of int
    :param data_type: Data type given to the model
    :type data_type: str
    :param config_mha: Multihead Attention configuration, allow reconstituion
    :type config_mha: dict
    """
    def __init__(self, input_shape, output_shape, data_type, config_mha = []) -> None:
        """Constructor."""
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.data_type = data_type
        self.config_mha = config_mha

    def create_mha_conf_equal(self, nb_mha, num_heads, embed_dim, dropout, batch_first):
        """Allow to easily create Multihead configuration if all are equals.

        :param nb_mha: Number of multihead in the model
        :type nb_mha: int
        :param num_heads: Number of parallel attention heads.
        :type num_heads: int
        :param embed_dim: Total dimension of the Multihead.
        :type embed_dim: int
        :param dropout: Dropout probability
        :type dropout: float
        :param batch_first: Define if yes or not the batch is first in the input and output tensor.
        :type batch_first: bool
        """
        self.config_mha = []
        for i in range(nb_mha):
            self.config_mha.append({'embed_dim': embed_dim, 'num_heads': num_heads, 'dropout': dropout, 'batch_first': batch_first})


    @classmethod
    def orbit5k(cls):
        config_mha = [{'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True},
                      {'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True},
                      {'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True},
                      {'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True},
                      {'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True}]
        return cls([4, 1290, 4], [4], "long", config_mha)
    
    @classmethod
    def orbit5kbig(cls):
        config_mha = [{'embed_dim': 128, 'num_heads': 32, 'dropout': 0.1, 'batch_first': True},
                      {'embed_dim': 128, 'num_heads': 32, 'dropout': 0.1, 'batch_first': True},
                      {'embed_dim': 128, 'num_heads': 32, 'dropout': 0.1, 'batch_first': True},
                      {'embed_dim': 128, 'num_heads': 32, 'dropout': 0.1, 'batch_first': True},
                      {'embed_dim': 128, 'num_heads': 32, 'dropout': 0.1, 'batch_first': True},
                      {'embed_dim': 128, 'num_heads': 32, 'dropout': 0.1, 'batch_first': True},
                      {'embed_dim': 128, 'num_heads': 32, 'dropout': 0.1, 'batch_first': True},
                      {'embed_dim': 128, 'num_heads': 32, 'dropout': 0.1, 'batch_first': True},
                      {'embed_dim': 128, 'num_heads': 32, 'dropout': 0.1, 'batch_first': True}]
        return cls([4, 1309, 4], [4], "long", config_mha)
    
    def video_transform(cls):
        config_mha = [{'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}, {'embed_dim': 1280, 'num_heads': 16, 'dropout': 0.0, 'batch_first': True}]
        return cls([4, 3, 518, 518], [4, 1000], "float", config_mha)

    