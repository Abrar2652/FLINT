# SCHEMA.md — The fused heterogeneous graph

This is the contract `src/flint/graph/builder.py` must satisfy. It defines the
PyG `HeteroData` object that one table (plus its candidates and the relevant
ontology neighborhood) is compiled into. Gate 2 tests check this schema.

## Node types

| Node type        | One node per …                              | Initial features |
|------------------|---------------------------------------------|------------------|
| `column`         | column of the table                         | header embedding (Sentence-Transformer), dtype flags, position |
| `cell_candidate` | (cell, candidate-entity) pair               | surface-similarity features, entity prior (log pagerank), entity description embedding |
| `onto_class`     | Wikidata class in the extracted neighborhood| class label embedding, depth-in-hierarchy |
| `onto_property`  | Wikidata property in the neighborhood       | property label embedding, domain/range type ids |

## Edge types (relation-typed; direction matters)

| Edge (src, rel, dst)                              | Meaning |
|---------------------------------------------------|---------|
| `(cell_candidate, in_column, column)`             | candidate belongs to this column |
| `(cell_candidate, same_row, cell_candidate)`      | two candidates in the same table row (enables n-ary / collective signal) |
| `(cell_candidate, instance_of, onto_class)`       | candidate's KG type(s) |
| `(onto_class, subclass_of, onto_class)`           | ontology hierarchy edge (carries hop distance as edge attr) |
| `(onto_class, has_property, onto_property)`       | property whose domain includes this class |
| `(cell_candidate, kg_triple, cell_candidate)`     | a Wikidata statement links these two candidates (carries property id as edge attr) |

> The `subclass_of` edge attribute `hop` and the `kg_triple` edge attribute
> `property_id` are what the **ontology-conditioned message function** reads to
> implement contribution #1. Do not drop them.

## Readout targets

- **CTA**: classification over `column` nodes -> distribution over `onto_class`.
- **CPA**: classification over ordered `column`-pairs -> distribution over `onto_property`
  (including a NULL "no relation" class).

## Invariants (enforced by tests/test_graph_builder.py)

1. Row order does not change the graph up to node permutation (permutation invariance).
2. Every `cell_candidate` has exactly one `in_column` edge.
3. The ontology subgraph is connected to candidates via at least one `instance_of` edge
   (no floating ontology island), unless the column has zero linkable cells.
4. `subclass_of` edges form a DAG (no cycles) within the extracted neighborhood.
5. Self-consistency: any `onto_property` reachable as a CPA target has its domain/range
   classes present as `onto_class` nodes.
