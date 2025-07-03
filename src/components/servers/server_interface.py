from abc import ABC, abstractmethod

class ServerI(ABC):

    @abstractmethod
    def init(self):
        raise RuntimeError("Must be implemented")
    
    @abstractmethod
    def shutdown(self):
        raise RuntimeError("Must be implemented")
    
    # @abstractmethod
    # def clear(self):
    #     raise RuntimeError("Must be implemented")
