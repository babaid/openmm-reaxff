#These are tools to handle some things for topologies, like naming for example
from openmm.app import Topology
from openmm import Vec3
class TopologyTools:
    @staticmethod
    def add_chain_name(chain_id:str, topology: Topology):
        for chain in topology.chains():
            if chain.id != "X":
                print("Warning, renaming chain, old chain id was ", chain.id, ".")
            chain.id = chain_id


def calculate_bounding_box(pdb):
    positions = pdb.positions
    
    min_x = min(pos.x for pos in positions)
    min_y = min(pos.y for pos in positions)
    min_z = min(pos.z for pos in positions)
    
    max_x = max(pos.x for pos in positions)
    max_y = max(pos.y for pos in positions)
    max_z = max(pos.z for pos in positions)
    
    return Vec3(min_x, min_y, min_z), Vec3(max_x, max_y, max_z)
