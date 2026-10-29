"""A general-purpose hypergraph with labelled, typed incidences.

A hypergraph has two kinds of thing:

    vertices    points (type V)
    edges       connections (type E, a unique key) that each join any number of
                vertices, and carry a payload (type P)

Each edge reaches its vertices through *roles* (type R). A role names what a
particular vertex is *to that edge*, so direction, order and meaning are never
ambiguous and an edge may touch the same vertex through several roles.

Examples of how other domains map onto it:

    ordinary graph        edge with roles {"from": a, "to": b}
    undirected graph      edge with roles {0: a, 1: b}
    flow network          edge {"from": a, "to": b}, payload = capacity
    Petri net             vertices = places + transitions; arcs as edges
    dependency / build    edge {"target": t, "dep": d1, ...}, one role per dep
    chemical reaction     vertices = species; edge {"reactant0": a, "product0": c}
    circuit               vertices = nets; edge {"p": n1, "n": n2}, payload = device

Requires Python 3.12+ (PEP 695 generic syntax).
"""
from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class Incidence[E: Hashable, R: Hashable]:
    """One attachment point: edge `edge` reaching a vertex through `role`."""
    edge: E
    role: R


class Hypergraph[V: Hashable, E: Hashable, R: Hashable, P]:
    def __init__(self) -> None:
        self._payloads: dict[E, P] = {}
        # edge -> (role -> vertex)
        self._links: dict[E, dict[R, V]] = {}
        # vertex -> incidences touching it. A dict (used as an ordered set) rather
        # than a set, so iteration order is deterministic between runs.
        self._vertices: dict[V, dict[Incidence[E, R], None]] = {}

    # ---- queries ---------------------------------------------------------

    def hasVertex(self, vertex: V) -> bool:
        return vertex in self._vertices

    def hasEdge(self, edge: E) -> bool:
        return edge in self._payloads

    def vertices(self) -> Iterator[V]:
        return iter(self._vertices)

    def edges(self) -> Iterator[tuple[E, P]]:
        return iter(self._payloads.items())

    def payload(self, edge: E) -> P:
        return self._payloads[edge]

    def links(self, edge: E) -> Mapping[R, V]:
        """Read-only view of {role: vertex} for an edge."""
        return MappingProxyType(self._links[edge])

    def vertexOf(self, edge: E, role: R) -> V:
        return self._links[edge][role]

    def incident(self, vertex: V) -> tuple[Incidence[E, R], ...]:
        """Every incidence touching this vertex (a snapshot, safe to mutate the graph while iterating)."""
        return tuple(self._vertices[vertex])

    def degree(self, vertex: V) -> int:
        return len(self._vertices[vertex])

    def neighbours(self, edge: E) -> Iterator[tuple[R, V, Incidence[E, R]]]:
        """For each role of `edge`, yield (that role, the vertex it reaches, each other incidence on that vertex)."""
        for role, vertex in self._links[edge].items():
            for other in tuple(self._vertices[vertex]):
                if other.edge != edge:
                    yield role, vertex, other

    # ---- mutation --------------------------------------------------------

    def addVertex(self, vertex: V) -> None:
        """Idempotent. Vertices are also created implicitly by addEdge."""
        self._vertices.setdefault(vertex, {})

    def removeVertex(self, vertex: V) -> None:
        """Only allowed once no edge reaches it (remove those edges first)."""
        if self._vertices[vertex]:
            raise ValueError(f"vertex {vertex!r} still has {len(self._vertices[vertex])} incidence(s)")
        del self._vertices[vertex]

    def addEdge(self, edge: E, payload: P, links: Mapping[R, V]) -> None:
        """Add an edge with all of its role -> vertex links in one step."""
        if edge in self._payloads:
            raise ValueError(f"edge {edge!r} already exists")
        if not links:
            raise ValueError(f"edge {edge!r} must link at least one vertex")

        for vertex in links.values():
            self.addVertex(vertex)

        self._payloads[edge] = payload
        self._links[edge] = dict(links)
        for role, vertex in links.items():
            self._vertices[vertex][Incidence(edge, role)] = None

    def removeEdge(self, edge: E) -> None:
        """Remove an edge. The vertices it touched are kept."""
        for role, vertex in self._links.pop(edge).items():
            del self._vertices[vertex][Incidence(edge, role)]
        del self._payloads[edge]
