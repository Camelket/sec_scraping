from abc import ABC
from typing import Any, Hashable
import logging
from collections import defaultdict
logger = logging.getLogger(__name__)


class SimpleDirectedGraph:
    '''
    A simple directed graph.
    '''
    def __init__(self):
        self._edge_data: dict[tuple[Hashable, Hashable], Hashable] = dict() #((from, to), relationship)
        self._connections: dict[Hashable, set[Hashable]] = defaultdict(set) # parent/child
        self._nodes: set[Hashable] = set()
    
    def connect_nodes(self, from_node: Hashable, to_node: Hashable, edge: str=None):
        for each in [from_node, to_node]:
            if each not in self._nodes:
                self._add_node(each)
        self._add_edge(from_node, to_node, edge)
    
    def _add_edge(self, parent: Hashable, child: Hashable, edge: Hashable|None):
        self._add_connection(parent, child)
        self._edge_data[(parent, child)] = edge
    
    def remove_edge(self, parent: Hashable, child: Hashable):
        if self._edge_data.get((parent, child), None):
            self._edge_data.popitem((parent, child))
            self._remove_relationship(parent, child)
        else:
            logger.info(f"No edge between ({parent}, {child}) could be found")
    
    def remove_all_edges(self, node: Hashable):
        for child in self._connections.get(node, []):
            for pair in [(node, child), (child, node)]:
                edge = self._edge_data.get(pair)
                if edge:
                    self._edge_data.popitem(pair)

    def _add_connection(self, parent: Hashable, child: Hashable):
        self._connections[parent].add(child)
    
    def _remove_relationship(self, parent: Hashable, child: Hashable):
        self._connections[parent].remove(child)
  

    def _add_node(self, node: Hashable):
        if node not in self._nodes:
            pass
        else:
            # warn that we are removing node
            logger.info(f"replacing an existing node with a new node: {node}")
            # remove old

        # add new
        self._nodes.add(node)
    
    def remove_node(self, node: tuple[int, int]):
        pass
    
    def disconnect_nodes(self, node1: Hashable, node2: Hashable):
        self.remove_edge(node1, node2)
        self.remove_edge(node2, node1)
        
        
    

class ReferenceSearch:
    '''
    reference search for different kinds of named entities.
    '''