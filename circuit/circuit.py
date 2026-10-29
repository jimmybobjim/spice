from collections.abc import Hashable, Iterable

from circuit.components import Component
from circuit.terminals import Terminals
from errors import CircuitError, SpiceError
from graph import Hypergraph

type Circuit[V: Hashable, E: Hashable] = Hypergraph[V, E, Terminals, Component]

def buildCircuit(components: Iterable[Component]) -> Circuit[str, str]:
    """Turn a flat list of devices into the graph the solver works on.

    Vertices are node names and edges are device names; each device reaches its nodes through its
    own terminal roles (``component.terminals``), in the order of ``component.nodes``.
    """
    graph: Circuit[str, str] = Hypergraph()
    for component in components:
        if graph.hasEdge(component.name):
            raise SpiceError(f"duplicate component name {component.name!r}", component.line)
        if len(component.nodes) != len(component.terminals):
            raise CircuitError(f"{component.name!r}: expected {len(component.terminals)} nodes, got {len(component.nodes)}")
        graph.addEdge(component.name, component, dict(zip(component.terminals, component.nodes)))
    return graph
