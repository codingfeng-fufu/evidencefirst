"""
Evidence Chain utilities for EvidenceFirst.

Converts between different graph representations and extracts entities.
"""

import re
import networkx as nx
import nltk
from typing import List, Tuple, Optional, Set

# Question words to filter (used in multiple places)
QUESTION_WORDS = {
    'what', 'when', 'where', 'who', 'which', 'whom', 'whose', 'why', 'how',
    'did', 'do', 'does', 'is', 'are', 'was', 'were', 'has', 'have', 'had',
    'can', 'could', 'will', 'would', 'should', 'may', 'might', 'must',
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'into', 'during', 'including',
    'both', 'either', 'neither', 'each', 'every', 'all', 'some', 'any',
    'more', 'most', 'less', 'least', 'many', 'much', 'few', 'several'
}


def normalize_entity(entity: str) -> str:
    """
    Normalize entity name for matching.

    Normalization rules:
    1. Lowercase
    2. Remove punctuation
    3. Remove leading articles (the, a, an)
    4. Strip whitespace
    5. Collapse multiple spaces

    Examples:
        "The Straight Story" -> "straight story"
        "St. Elizabeths Hospital" -> "st elizabeths hospital"
        "Christopher Nolan" -> "christopher nolan"
    """
    if not entity:
        return ""

    # Lowercase
    normalized = entity.lower()

    # Remove punctuation except spaces
    normalized = re.sub(r'[^\w\s]', '', normalized)

    # Remove leading articles
    for article in ['the ', 'a ', 'an ']:
        if normalized.startswith(article):
            normalized = normalized[len(article):]
            break

    # Collapse multiple spaces and strip
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized


def find_entity_in_graph(entity: str, graph_nodes: List[str], threshold: float = 0.85) -> Optional[str]:
    """
    Find matching entity in graph nodes using normalization and fuzzy matching.

    Args:
        entity: Entity name to find
        graph_nodes: List of node names in the graph
        threshold: Similarity threshold for fuzzy matching (0-1)

    Returns:
        Matching node name from graph, or None if no match found
    """
    if not entity or not graph_nodes:
        return None

    normalized_entity = normalize_entity(entity)

    # Strategy 1: Exact match after normalization
    for node in graph_nodes:
        if normalize_entity(node) == normalized_entity:
            return node

    # Strategy 2: Partial match (entity is substring or vice versa)
    for node in graph_nodes:
        normalized_node = normalize_entity(node)
        if normalized_entity in normalized_node or normalized_node in normalized_entity:
            # Only accept if length difference is reasonable
            if abs(len(normalized_entity) - len(normalized_node)) <= max(len(normalized_entity), len(normalized_node)) * 0.3:
                return node

    # Strategy 3: Fuzzy matching using simple token overlap
    entity_tokens = set(normalized_entity.split())
    best_match = None
    best_score = 0.0

    for node in graph_nodes:
        node_tokens = set(normalize_entity(node).split())
        if not entity_tokens or not node_tokens:
            continue

        # Jaccard similarity
        intersection = len(entity_tokens & node_tokens)
        union = len(entity_tokens | node_tokens)
        score = intersection / union if union > 0 else 0

        if score > best_score and score >= threshold:
            best_score = score
            best_match = node

    return best_match

# Reuse NLTK entity extraction
def extract_entities_nltk(text: str) -> List[str]:
    """Extract named entities + noun phrases via NLTK."""
    entities = []
    try:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        tree = nltk.ne_chunk(tagged)

        # Extract NER chunks
        for subtree in tree:
            if hasattr(subtree, 'label'):
                phrase = " ".join(tok for tok, _ in subtree.leaves())
                entities.append(phrase)

        # Extract noun phrase chunks (improved: only proper nouns)
        chunk = []
        for tok, pos in tagged:
            # Skip question words even if tagged as proper noun
            if tok.lower() in QUESTION_WORDS:
                if len(chunk) >= 1:
                    entities.append(" ".join(chunk))
                chunk = []
                continue

            if pos in ('NNP', 'NNPS'):  # Only proper nouns
                chunk.append(tok)
            else:
                if len(chunk) >= 1:  # Accept single words too
                    entities.append(" ".join(chunk))
                chunk = []
        if len(chunk) >= 1:
            entities.append(" ".join(chunk))

    except Exception:
        # Fallback: only capitalized words (excluding question words)
        entities = []
        words = text.split()
        chunk = []
        for w in words:
            if w.lower() in QUESTION_WORDS:
                if chunk:
                    entities.append(" ".join(chunk))
                    chunk = []
                continue
            if w and w[0].isupper():
                chunk.append(w)
            else:
                if chunk:
                    entities.append(" ".join(chunk))
                    chunk = []
        if chunk:
            entities.append(" ".join(chunk))

    return list(set(entities)) if entities else []


def parse_triple_string(triple_str: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse "A -- relation --> B" format to (head, relation, tail).

    Examples:
        "Christopher Nolan -- nationality --> British"
        → ("Christopher Nolan", "nationality", "British")

    Returns:
        (head, relation, tail) or None if parsing fails
    """
    if not triple_str or not isinstance(triple_str, str):
        return None

    # Pattern: "head -- relation --> tail"
    match = re.match(r"^(.+?)\s*--\s*(.+?)\s*-->\s*(.+)$", triple_str.strip())
    if match:
        return (
            match.group(1).strip(),
            match.group(2).strip(),
            match.group(3).strip()
        )

    # Fallback: split by "--" and ">"
    if "--" in triple_str and ">" in triple_str:
        parts = triple_str.split("--")
        if len(parts) >= 2:
            head = parts[0].strip()
            rest = "--".join(parts[1:])
            if "-->" in rest:
                rel_tail = rest.split("-->")
                if len(rel_tail) == 2:
                    relation = rel_tail[0].strip()
                    tail = rel_tail[1].strip()
                    return (head, relation, tail)

    return None


def build_graph_from_triple_strings(triple_strings: List[str]) -> nx.Graph:
    """
    Build an undirected NetworkX graph from triple string list.

    Args:
        triple_strings: List of "head -- relation --> tail" strings

    Returns:
        NetworkX Graph with edges having 'relation' attribute
    """
    G = nx.Graph()

    for triple_str in triple_strings:
        parsed = parse_triple_string(triple_str)
        if parsed:
            head, relation, tail = parsed
            # Add edge (undirected, so order doesn't matter)
            G.add_edge(head, tail, relation=relation)

    return G


def convert_digraph_to_graph(G_directed: nx.DiGraph) -> nx.Graph:
    """
    Convert a directed graph (with relations list) to undirected graph
    (with single relation per edge).

    Args:
        G_directed: nx.DiGraph where edges have data={"relations": [...]}

    Returns:
        nx.Graph where edges have data={"relation": "..."}
    """
    G_undirected = nx.Graph()

    for u, v, data in G_directed.edges(data=True):
        relations = data.get("relations", [])
        # Take first relation or default
        relation = relations[0] if relations else "related_to"

        # If edge already exists (from reverse direction), keep existing
        if not G_undirected.has_edge(u, v):
            G_undirected.add_edge(u, v, relation=relation)

    return G_undirected


def extract_question_entities(question: str) -> List[str]:
    """
    Extract entities from the question.

    These serve as the starting points for evidence chain verification.

    Always returns at least one entity (fallback to whole question if needed).
    """
    # Clean punctuation from question first
    cleaned_question = re.sub(r'[?.,;:!]', '', question)

    entities = extract_entities_nltk(cleaned_question)

    # Filter out question words and clean punctuation
    filtered_entities = []
    for e in entities:
        # Remove trailing/leading punctuation
        e_clean = e.strip('?.,;:!')
        # Check if it's not a question word
        if e_clean.lower() not in QUESTION_WORDS:
            filtered_entities.append(e_clean)

    entities = filtered_entities

    # Fallback 1: if no entities found, use sequences of capitalized words
    if not entities:
        words = cleaned_question.split()
        # Filter out question words (case-insensitive)
        words = [w for w in words if w and w[0].isupper() and w.lower() not in QUESTION_WORDS]
        if words:
            # Group consecutive capitalized words as entities
            entities = [" ".join(words)]

    # Fallback 2: if still empty, use the whole cleaned question
    if not entities:
        entities = [cleaned_question]

    return entities


def select_candidate_answers(graph: nx.Graph,
                             question_entities: List[str],
                             strategy: str = "all") -> List[str]:
    """
    Select candidate answer entities from the graph.

    Args:
        graph: Retrieved KG subgraph
        question_entities: Entities from the question (starting points)
        strategy: "all" | "distant" | "leaf"

    Returns:
        List of candidate answer entity names (never empty - fallback to all nodes if needed)
    """
    if graph.number_of_nodes() == 0:
        return []

    all_nodes = list(graph.nodes())

    # Always exclude question entities from candidates
    question_entities_set = set(question_entities)

    if strategy == "all":
        # Simple: all nodes except question entities
        candidates = [n for n in all_nodes if n not in question_entities_set]
        # Fallback: if all nodes are question entities, use all nodes anyway
        return candidates if candidates else all_nodes

    elif strategy == "distant":
        # Only nodes that are 2-3 hops away from question entities
        candidates = set()
        for q_entity in question_entities:
            if q_entity in graph.nodes():
                # BFS to find nodes at distance 2-3
                distances = nx.single_source_shortest_path_length(
                    graph, q_entity, cutoff=3
                )
                for node, dist in distances.items():
                    if 2 <= dist <= 3:
                        candidates.add(node)
        candidates -= question_entities_set
        # Fallback: if no distant candidates, use all non-question nodes
        if not candidates:
            candidates = [n for n in all_nodes if n not in question_entities_set]
        # Final fallback: if still empty, use all nodes
        return list(candidates) if candidates else all_nodes

    elif strategy == "leaf":
        # Nodes with degree 1 (leaf nodes), excluding question entities
        candidates = [n for n in all_nodes if graph.degree(n) == 1 and n not in question_entities_set]
        # Fallback chain
        if not candidates:
            candidates = [n for n in all_nodes if n not in question_entities_set]
        if not candidates:
            candidates = all_nodes
        return candidates

    # Default: all nodes except question entities, with fallback
    candidates = [n for n in all_nodes if n not in question_entities_set]
    return candidates if candidates else all_nodes


def extract_evidence_chain_triples(graph: nx.Graph,
                                    path: List[str]) -> List[Tuple[str, str, str]]:
    """
    Extract triples along a path in the graph.

    Args:
        graph: NetworkX graph
        path: List of node names forming a path

    Returns:
        List of (head, relation, tail) triples
    """
    triples = []
    for i in range(len(path) - 1):
        head = path[i]
        tail = path[i + 1]
        edge_data = graph.get_edge_data(head, tail)
        relation = edge_data.get('relation', 'related_to') if edge_data else 'related_to'
        triples.append((head, relation, tail))
    return triples


def triples_to_strings(triples: List[Tuple[str, str, str]]) -> List[str]:
    """
    Convert (head, relation, tail) tuples back to string format.

    Returns:
        List of "head -- relation --> tail" strings
    """
    return [f"{h} -- {r} --> {t}" for h, r, t in triples]


# ============================================================
# Unit Tests
# ============================================================

def test_parse_triple_string():
    """Test triple string parsing."""
    # Normal case
    result = parse_triple_string("Christopher Nolan -- nationality --> British")
    assert result == ("Christopher Nolan", "nationality", "British")

    # With extra spaces
    result = parse_triple_string("  A  --  rel  -->  B  ")
    assert result == ("A", "rel", "B")

    # Invalid format
    result = parse_triple_string("not a triple")
    assert result is None

    print("✓ test_parse_triple_string passed")


def test_build_graph():
    """Test graph building from strings."""
    triples = [
        "A -- rel1 --> B",
        "B -- rel2 --> C",
        "C -- rel3 --> D"
    ]

    graph = build_graph_from_triple_strings(triples)

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3
    assert graph.has_edge("A", "B")
    assert graph.has_edge("B", "C")

    # Check relation attribute
    edge_data = graph.get_edge_data("A", "B")
    assert edge_data["relation"] == "rel1"

    print("✓ test_build_graph passed")


def test_entity_extraction():
    """Test entity extraction."""
    question = "What is the capital of the country where Christopher Nolan was born?"
    entities = extract_question_entities(question)

    # Check that at least some entities were extracted
    assert len(entities) > 0, f"No entities extracted from: {question}"
    print(f"✓ test_entity_extraction passed: extracted {len(entities)} entities: {entities}")


def test_candidate_selection():
    """Test candidate answer selection."""
    triples = [
        "Question Entity -- rel1 --> Mid1",
        "Mid1 -- rel2 --> Mid2",
        "Mid2 -- rel3 --> Answer Entity"
    ]
    graph = build_graph_from_triple_strings(triples)

    # All strategy - should exclude question entity
    candidates = select_candidate_answers(graph, ["Question Entity"], strategy="all")
    assert len(candidates) == 3  # 4 nodes - 1 question entity = 3
    assert "Question Entity" not in candidates

    # Distant strategy
    candidates = select_candidate_answers(graph, ["Question Entity"], strategy="distant")
    assert "Mid2" in candidates or "Answer Entity" in candidates

    # Edge case: all nodes are question entities
    candidates = select_candidate_answers(graph, list(graph.nodes()), strategy="all")
    assert len(candidates) == 4  # Fallback to all nodes

    print("✓ test_candidate_selection passed")


if __name__ == "__main__":
    print("Running evidence_utils unit tests...\n")
    test_parse_triple_string()
    test_build_graph()
    test_entity_extraction()
    test_candidate_selection()
    print("\n✅ All evidence_utils tests passed!")
