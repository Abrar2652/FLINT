# DATA_SPEC.md — CEA Candidate Generation & Distant-Supervision Labeling

> The two hardest engineering pieces, and the ones GRAMS+ spends the most effort
> on. This doc specifies (A) how cells get candidate entities and scores (CEA),
> which feed the graph builder, and (B) how Wikipedia tables become training labels
> via distant supervision. Both must be reproducible and leakage-safe. Read
> `docs/SCHEMA.md` and `docs/MODEL_SPEC.md` first — those define what consumes this.

---

## Part A — CEA: candidate generation & scoring

The graph builder (SCHEMA.md) needs, for each cell, a set of candidate entities
with features. CEA is upstream of CTA/CPA and is NOT a paper contribution — so we
follow GRAMS+ closely and keep it boring and reproducible.

### A.1 Detect entity columns
A column is "linkable" if the majority of its cells are classified as text
(spaCy NER/POS + regex for numbers/dates), matching GRAMS+'s heuristic (high
recall ~0.75-0.8 precision is acceptable — the model tolerates noise). Output:
`entity_columns: list[int]`.

### A.2 Retrieve candidates (union of sources, capped at K)
For each cell in an entity column, union candidates from:
1. Wikidata search API (`wbsearchentities`),
2. ElasticSearch keyword search over a local Wikidata label index,
3. fuzzy name match (SymSpell / Generic-Jaccard over the label index).

Cap at **K = 100** candidates/cell (GRAMS+'s setting; Fig. 4 shows our method
improves as K grows, so K is also a robustness knob — sweep `{1,3,5,10,20,50,100}`).

> Implementation note: build the local Wikidata label + pagerank index ONCE from
> the 2023-06-19 dump (DATASET_FACTS.md) and cache it. Do not hit the live API at
> train/eval time — it is non-reproducible and rate-limited. The live API is only
> for an optional ablation on retrieval source.

### A.3 Candidate features (consumed as cell_candidate node features, SCHEMA.md)
Per (cell, candidate):
- **Surface**: Levenshtein, Jaro-Winkler, Monge-Elkan, Generic-Jaccard between
  mention and candidate label (4 scalars).
- **Context**: weighted dot product of SentenceTransformer embeddings of (column
  header) and (candidate description); count of other cells matched by this
  candidate's properties / 20 (rescale). (2 scalars + the embedding itself.)
- **Prior**: normalized log pagerank (Eq. in GRAMS+):
  `(log pr(e) - min) / (max - min)` over the dump. (1 scalar.)
- **Header-absent variant**: also store features with header masked, so the
  header-dropout robustness perturbation (perturbations.py) is supported without
  recomputation.

### A.4 Interface (wire into data/loaders.py + graph/builder.py)
```python
@dataclass
class Candidate:
    entity_id: str
    label: str
    type_ids: list[str]        # instance_of targets -> instance_of edges
    surface_feats: dict        # 4 scalars
    context_feats: dict        # 2 scalars + embedding handle
    prior: float               # normalized log pagerank
    statements: list[tuple]    # (property_id, object_entity_id) -> kg_triple edges

def generate_candidates(record, kg_index, encoder, k=100) -> dict:
    """record -> {(row, col): list[Candidate]} for cols in record.entity_columns."""
```

---

## Part B — Distant-supervision labeling (the training-data engine)

Turns raw Wikipedia tables into (graph, CTA-gold, CPA-gold) WITHOUT manual
annotation, by trusting in-table hyperlinks as entity ground truth. This is what
makes the method work without a labeled corpus — and its noise is the reason
contributions #2/#3 (calibration/robustness) matter.

### B.1 "Easy-to-label" table filter (GRAMS+'s conditions — replicate exactly)
Keep a Wikipedia table only if: (1) >= 10 rows; (2) <= 1 hyperlink per cell;
(3) some column has >= 70% cells hyperlinked; (4) > 80% of links resolve to a
Wikidata entity. These four are encoded as asserts so a reviewer can audit the
filter. Sample **5,000** tables for training (GRAMS+'s size).

### B.2 Context-inconsistent-link removal (the blocklist)
Some columns link to the wrong type (the "city column linking to airport" case).
Procedure:
1. Assign each column a provisional type = most common type of its linked entities.
2. Build a blocklist by reviewing **normalized headers** (mask numbers, strip
   punctuation) that appear across multiple provisional types; label incompatible
   (header, type) pairs. GRAMS+ reviewed ~230 headers — ship the resulting
   blocklist as a versioned data file `data/processed/blocklist.tsv`, NOT code.
3. Drop links in a column whose header is incompatible with its provisional type.

### B.3 CTA gold from links
After cleaning, a column's CTA label = the type selected by the SAME greedy
ancestor procedure the model is meant to learn — BUT here it is a *labeling*
heuristic, used only to generate training targets, and it runs on the
*hyperlink-resolved* (high-confidence) entities, not noisy candidates. Keep this
labeler in `data/distant_supervision.py`, clearly separated from the model.

### B.4 CPA gold from matched statements (the part that's genuinely fiddly)
1. For each row, for each linked entity, match its Wikidata statement values
   against other cells in the same row -> candidate relationships (incl. n-ary via
   statement nodes), per MODEL_SPEC/SCHEMA `kg_triple` edges.
2. Group candidate relationships by (source col, target col / literal).
3. Drop a relationship if it occurs in **< 50%** of rows OR if **> 10%** of rows
   show data contradicting the KG. (GRAMS+'s thresholds.)
4. Among survivors per column-pair, keep the **highest-frequency** property as
   CPA gold. Generated-but-unselected relationships become **negative** examples.

### B.5 Leakage guard (NON-NEGOTIABLE, CLAUDE.md s2)
Before emitting training data, call
`data.loaders.assert_no_250wt_leakage(train_article_ids, wt250_article_ids)`.
Any 250WT source article in the training set is a hard failure. Also exclude by
Wikidata/Wikipedia dump dates in DATASET_FACTS.md so eval entities aren't
time-leaked.

### B.6 Noise is a feature, not just a bug
Record, per generated label, a confidence proxy (e.g., row-frequency of the
relationship, fraction of links resolved). This per-label confidence is (a) a
useful training-time loss weight, and (b) the natural quantity to study in the
calibration experiment — "does the model's conformal coverage hold even though
training labels were noisy?" is a strong robustness sub-result.

### B.7 Interface (wire into data/distant_supervision.py)
```python
def is_easy_to_label(wiki_table) -> bool: ...                 # B.1
def clean_inconsistent_links(table, blocklist) -> table: ...  # B.2
def label_cta(table, kg_index) -> dict[int, str]: ...         # B.3
def label_cpa(table, kg_index) -> tuple[dict, list]: ...      # B.4 (gold, negatives)
def build_training_examples(wiki_dump, kg_index, n=5000, *, wt250_ids) -> list[STIRecord]:
    """Full pipeline B.1->B.5, asserts leakage guard, attaches B.6 confidences."""
```

---

## Part C — Determinism & caching (both parts)
- The Wikidata label/pagerank index and candidate sets are cached to
  `data/cache/` keyed by (dump date, K, source set). Training never recomputes them.
- Candidate retrieval given a fixed cache is deterministic; record the cache hash
  in each run dir (CLAUDE.md s2 config-as-artifact).
- The 5,000-table training sample is drawn with a fixed seed and the chosen
  table_ids written to `data/processed/train_table_ids.txt` so the exact set is reproducible.

## Part D — Gate mapping
- Part A.1-A.3 + the cache: needed for **Gate 2** (graph builder has real candidates).
- Part B (all): needed for **Gate 3** (model has training targets).
- Part B.6 confidence: feeds the **Gate 4** calibration analysis.
- A.2 header-absent variant + K-sweep: feed the **Gate 5** robustness curves.
