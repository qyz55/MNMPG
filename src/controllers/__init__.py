REGISTRY = {}

from .basic_controller import BasicMAC
from .separate_controller import SeparateMAC
from .hierarchical_controller import HierarchicalMAC
from .robust_controller import RobustMAC
from .hierarchical_rode_controller import HierarchicalRODEMAC
from .hierarchical_noise_controller import HierarchicalNoiseMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["separate_mac"]=SeparateMAC
REGISTRY["hierarchical_mac"]=HierarchicalMAC
REGISTRY["robust_mac"]=RobustMAC
REGISTRY["hierarchical_rode_mac"]=HierarchicalRODEMAC
REGISTRY["hierarchical_noise_mac"]=HierarchicalNoiseMAC