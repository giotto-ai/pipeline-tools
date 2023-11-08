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

def BToMb(x):
    """Convert bytes to megabytes.
    
    :param x: Value in bytes.
    :type x: int
    :return: Value in megabytes.
    :rtype: int
    """
    return (x // (2 * 1024))

class TraceMalloc():
    def __init__(self, nb_gpu):
        """Initialize a TraceMalloc object.
        
        :param nb_gpu: Number of GPUs.
        :type nb_gpu: int
        """
        self.nb_gpu = nb_gpu
        self.begin  = [0] * nb_gpu
        self.end    = [0] * nb_gpu
        self.peak   = [0] * nb_gpu
        self.peaked = [0] * nb_gpu

    def __enter__(self):
        """Enter the context manager. 
        
        Save the current memory allocated to all GPUs.
        
        :return: The TraceMalloc object.
        :rtype: TraceMalloc
        """
        for device in range(self.nb_gpu):
            self.begin[device] = torch.cuda.memory_allocated(device)
            
        return self
    
    def __exit__(self, *exc):
        """Exit the context manager. 

        Get all the memory information, allocated and peak, to calculate the true peak between the enter and exit call.
        """
        for device in range(self.nb_gpu):
            self.end[device]    = torch.cuda.memory_allocated(device)
            self.peak[device]   = torch.cuda.max_memory_allocated(device)
            self.peaked[device] = BToMb(self.peak[device] - self.begin[device])
            torch.cuda.reset_peak_memory_stats(device)