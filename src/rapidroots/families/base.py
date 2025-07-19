import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod

class FunctionFamily(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique key under which this family is registered"""


    @property
    @abstractmethod
    def param_count(self) -> int:
        """Number of parametes this family takes (not counting x)"""

    @abstractmethod
    def evaluate(self, x: npt.NDArray, params: npt.NDArray, deravitive: int=0) -> npt.NDArray:
        """
        Compute f, f' or f'' on 'x' with shape (n,) and params with shape (n, param_count)"""