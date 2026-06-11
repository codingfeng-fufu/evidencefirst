"""
Evidence Chain Checker Module

Core innovation of EvidenceFirst: Pre-generation structural verification
that checks evidence chain completeness using graph connectivity analysis.

Key components:
1. Evidence Chain Completeness Check: Verify if a connected path exists
2. Bridge Gap Detection: Locate missing entities/relations in the chain
3. Targeted Gap Repair: Retrieve specific triples to bridge the gap
"""

import networkx as nx
import re
from typing import List, Dict, Tuple, Optional, Set, Union
from loguru import logger

# Import utilities for string conversion
from evidence_utils import (
    build_graph_from_triple_strings,
    extract_question_entities,
    find_entity_in_graph,
    select_candidate_answers,
    extract_evidence_chain_triples,
    triples_to_strings
)


def _minimum_chain_edges(question: str) -> int:
    """Minimum structural support expected before calling a chain complete."""
    q = str(question or "").lower().strip()
    yesno_prefixes = (
        "are ", "is ", "was ", "were ",
        "do ", "does ", "did ",
        "has ", "have ", "had ",
        "can ", "could ", "will ", "would ",
    )
    if q.startswith(yesno_prefixes):
        return 1
    return 2


_GENERIC_QUESTION_ENTITIES = {
    "american",
    "british",
    "president",
    "general",
    "film",
    "films",
    "city",
    "country",
    "location",
    "profession",
    "award",
}


def _normalize_entity_key(entity: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(entity or "").lower())).strip()


def _filter_question_entities(entities: List[str]) -> List[str]:
    filtered = []
    for entity in entities:
        key = _normalize_entity_key(entity)
        if not key or key in _GENERIC_QUESTION_ENTITIES:
            continue
        filtered.append(entity)
    return filtered or entities


def _infer_answer_type(question: str) -> str:
    """Infer a coarse answer type from the question surface form."""
    q = str(question or "").lower().strip()
    if any(marker in q for marker in (
        "which film is newer", "which film is older", "which is newer",
        "which is older", "who was born first", "who was born earlier",
        "who was born later", "released first", "came first",
    )):
        return "comparison"
    if " or " in q and q.startswith((
        "are ", "is ", "was ", "were ",
        "do ", "does ", "did ",
        "has ", "have ", "had ",
        "can ", "could ", "will ", "would ",
    )):
        return "choice"
    if q.startswith((
        "are ", "is ", "was ", "were ",
        "do ", "does ", "did ",
        "has ", "have ", "had ",
        "can ", "could ", "will ", "would ",
    )):
        return "yesno"
    if "award" in q or "prize" in q:
        return "award"
    if "profession" in q or "occupation" in q or "job" in q:
        return "profession"
    if q.startswith("where") or "what location" in q or "what city" in q or "which city" in q or "located" in q:
        return "location"
    if "what country" in q or "which country" in q or "country" in q:
        return "country"
    if "what year" in q or "when " in q or "what date" in q or "which period" in q or "during which period" in q:
        return "date"
    if "which president" in q or "what president" in q:
        return "ordinal"
    if "how many" in q or "population" in q or "number of" in q:
        return "number"
    if q.startswith("who ") or "which person" in q:
        return "person"
    return "entity"


def _node_matches_answer_type(node: str, answer_type: str) -> bool:
    text = str(node or "").lower()
    if not text:
        return False
    if answer_type in {"entity", "yesno", "choice", "comparison"}:
        return True
    if answer_type == "award":
        return any(marker in text for marker in (
            "award", "prize", "egot", "oscar", "emmy", "grammy", "tony",
            "academy", "golden globe", "pulitzer",
        ))
    if answer_type == "profession":
        return any(marker in text for marker in (
            "actor", "artist", "author", "composer", "director", "musician",
            "pianist", "player", "politician", "professor", "singer",
            "writer", "wrestler", "journalist", "footballer", "coach",
        ))
    if answer_type == "location":
        return (
            "," in text
            or any(marker in text for marker in (
                "city", "county", "state", "province", "district", "kingdom",
                "republic", "usa", "united states", "new york", "london",
            ))
        )
    if answer_type == "country":
        return any(marker in text for marker in (
            "country", "kingdom", "republic", "united states", "usa", "uk",
            "british", "american", "canada", "china", "france", "germany",
            "india", "japan", "russia", "yemen",
        ))
    if answer_type == "date":
        return bool(re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", text)) or any(
            month in text
            for month in (
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december",
            )
        )
    if answer_type == "number":
        return bool(re.search(r"\d", text))
    if answer_type == "ordinal":
        return bool(re.search(r"\b\d+(st|nd|rd|th)?\b", text))
    if answer_type == "person":
        return len(str(node).split()) >= 2 and "," not in text
    return True


def _filter_candidates_by_answer_type(
    candidate_answers: List[str],
    answer_type: str,
) -> List[str]:
    if answer_type in {"entity", "yesno", "choice", "comparison"}:
        return candidate_answers
    filtered = [
        candidate
        for candidate in candidate_answers
        if _node_matches_answer_type(candidate, answer_type)
    ]
    return filtered or candidate_answers


def _filter_candidates_by_min_distance(
    graph: nx.Graph,
    question_entities: List[str],
    candidate_answers: List[str],
    min_edges: int,
) -> List[str]:
    """Avoid treating immediate bridge nodes as final answer candidates."""
    if min_edges <= 1 or graph.number_of_nodes() == 0:
        return candidate_answers

    graph_nodes = list(graph.nodes())
    mapped_questions = [
        find_entity_in_graph(entity, graph_nodes, threshold=0.85) or entity
        for entity in question_entities
    ]

    filtered = []
    for candidate in candidate_answers:
        if candidate not in graph:
            continue
        for question_entity in mapped_questions:
            if question_entity not in graph:
                continue
            try:
                if nx.shortest_path_length(graph, question_entity, candidate) >= min_edges:
                    filtered.append(candidate)
                    break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    return filtered or candidate_answers


class EvidenceChainChecker:
    """
    Verifies evidence chain completeness before generation.

    Unlike post-hoc answer verification (Self-RAG, FAIR-RAG) or LLM-based
    quality assessment (CRAG, GraphAnchor), we use deterministic graph
    connectivity analysis to verify that retrieved evidence forms a complete
    reasoning path.
    """

    def __init__(self):
        self.stats = {
            'total_checks': 0,
            'complete_chains': 0,
            'incomplete_chains': 0,
            'gaps_detected': 0,
            'repairs_attempted': 0,
            'repairs_successful': 0
        }
        self.last_answer_type = ""

    def check_evidence_chain(
        self,
        sub_graph: Union[nx.Graph, List[str]],
        question_entities: Optional[List[str]] = None,
        candidate_answers: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if evidence chain is complete.

        Args:
            sub_graph: Retrieved KG subgraph as NetworkX graph OR list of triple strings
            question_entities: Entities mentioned in the question (optional, will extract if not provided)
            candidate_answers: Candidate answer entities from retrieval (optional, will select if not provided)

        Returns:
            (is_complete, gap_info)
            - is_complete: True if a connected path exists
            - gap_info: If incomplete, contains gap location details

        Core Innovation: This is pre-generation structural verification,
        not post-hoc LLM judgment. We verify evidence BEFORE generating answer.
        """
        self.stats['total_checks'] += 1

        # Convert triple strings to graph if needed
        if isinstance(sub_graph, list):
            sub_graph = build_graph_from_triple_strings(sub_graph)

        # Auto-extract question entities if not provided
        if question_entities is None:
            logger.warning("question_entities not provided, cannot check evidence chain")
            return False, {'gap_type': 'no_question_entities'}

        # Auto-select candidate answers if not provided
        if candidate_answers is None:
            candidate_answers = select_candidate_answers(sub_graph, question_entities, strategy="all")

        # Handle empty inputs
        if not question_entities or not candidate_answers:
            logger.warning("Empty question_entities or candidate_answers")
            return False, {
                'gap_type': 'empty_input',
                'question_entities': question_entities,
                'candidate_answers': candidate_answers
            }

        # Import fuzzy matching function
        from evidence_utils import find_entity_in_graph

        graph_nodes = list(sub_graph.nodes())

        # Map question entities to graph nodes using fuzzy matching
        mapped_question_entities = []
        for entity in question_entities:
            matched_node = find_entity_in_graph(entity, graph_nodes, threshold=0.85)
            if matched_node:
                mapped_question_entities.append(matched_node)
            else:
                # Keep original if no match found
                mapped_question_entities.append(entity)

        # Map candidate answers to graph nodes
        mapped_candidate_answers = []
        for entity in candidate_answers:
            matched_node = find_entity_in_graph(entity, graph_nodes, threshold=0.85)
            if matched_node:
                mapped_candidate_answers.append(matched_node)
            else:
                mapped_candidate_answers.append(entity)

        # Check if all mapped entities exist in graph
        missing_entities = []
        for entity in mapped_question_entities + mapped_candidate_answers:
            if entity not in sub_graph.nodes():
                missing_entities.append(entity)

        # Always calculate disconnected pairs (for B solution)
        disconnected_pairs = []
        for q_entity in mapped_question_entities:
            if q_entity not in sub_graph.nodes():
                continue
            for answer_entity in mapped_candidate_answers:
                if answer_entity not in sub_graph.nodes():
                    continue
                if not nx.has_path(sub_graph, q_entity, answer_entity):
                    disconnected_pairs.append((q_entity, answer_entity))

        if missing_entities:
            logger.debug(f"Missing entities in graph (after fuzzy matching): {missing_entities}")
            return False, {
                'gap_type': 'missing_entities',
                'missing': missing_entities,
                'original_question_entities': question_entities,
                'mapped_question_entities': mapped_question_entities,
                'disconnected_pairs': disconnected_pairs  # Include for A+B solution
            }

        # Check connectivity: Does a path exist?
        for q_entity in mapped_question_entities:
            for answer_entity in mapped_candidate_answers:
                if nx.has_path(sub_graph, q_entity, answer_entity):
                    # Complete chain found!
                    self.stats['complete_chains'] += 1
                    path = nx.shortest_path(sub_graph, q_entity, answer_entity)
                    logger.debug(f"Complete evidence chain found: {path}")
                    return True, None

        # No complete path found - detect gap
        self.stats['incomplete_chains'] += 1

        gap_info = self._locate_bridge_gap(
            sub_graph,
            mapped_question_entities,
            mapped_candidate_answers
        )
        gap_info['disconnected_pairs'] = disconnected_pairs

        return False, gap_info

    def _locate_bridge_gap(
        self,
        sub_graph: nx.Graph,
        start_entities: List[str],
        end_entities: List[str]
    ) -> Dict:
        """
        Locate where the evidence chain breaks.

        Key Innovation: Not just "chain is incomplete" (FAIR-RAG), but
        "chain breaks between entity X and entity Y" - enabling targeted repair.

        Returns:
            gap_info with keys:
            - gap_type: 'disconnected' / 'partial_path'
            - gap_start: Entity on the start side
            - gap_end: Entity on the end side
            - explanation: Human-readable diagnostic
        """
        self.stats['gaps_detected'] += 1

        # Find which start entity gets closest to which end entity
        best_gap = None
        min_distance = float('inf')

        for start in start_entities:
            if start not in sub_graph.nodes():
                continue

            # Use BFS to find reachable nodes from start
            reachable = nx.single_source_shortest_path_length(sub_graph, start)

            for end in end_entities:
                if end not in sub_graph.nodes():
                    continue

                if end in reachable:
                    # There IS a path - shouldn't happen, but handle it
                    continue

                # Find the closest reachable node to end
                # (This is a heuristic - in practice we'd do bidirectional search)
                end_neighbors = list(sub_graph.neighbors(end)) if end in sub_graph.nodes() else []

                for neighbor in end_neighbors:
                    if neighbor in reachable:
                        distance = reachable[neighbor]
                        if distance < min_distance:
                            min_distance = distance
                            best_gap = {
                                'gap_type': 'bridge_gap',
                                'gap_start': neighbor,  # Last reachable node from start
                                'gap_end': end,  # First node on end side
                                'distance': distance,
                                'explanation': f"Gap between '{neighbor}' and '{end}'"
                            }

        if best_gap:
            logger.debug(f"Bridge gap detected: {best_gap['explanation']}")
            return best_gap

        # Completely disconnected - no neighbors in common
        return {
            'gap_type': 'disconnected',
            'gap_start': start_entities[0] if start_entities else None,
            'gap_end': end_entities[0] if end_entities else None,
            'explanation': 'Question and answer entities are completely disconnected'
        }

    def targeted_gap_repair(
        self,
        full_kg: nx.Graph,
        gap_info: Dict,
        top_k: int = 10
    ) -> List[Tuple[str, str, str]]:
        """
        Retrieve triples to bridge the detected gap.

        Key Innovation: Targeted repair vs global re-retrieval (FAIR-RAG, CoRAG).
        We only retrieve triples connecting the two gap endpoints, not all evidence.

        Args:
            full_kg: Full knowledge graph (not just retrieved subgraph)
            gap_info: Gap location from _locate_bridge_gap
            top_k: Max number of bridge triples to retrieve

        Returns:
            List of (head, relation, tail) triples that bridge the gap
        """
        self.stats['repairs_attempted'] += 1

        gap_start = gap_info.get('gap_start')
        gap_end = gap_info.get('gap_end')

        if not gap_start or not gap_end:
            logger.warning("Cannot repair: gap_start or gap_end is None")
            return []

        # Strategy 1: Direct edge
        bridge_triples = []
        if full_kg.has_edge(gap_start, gap_end):
            edge_data = full_kg.get_edge_data(gap_start, gap_end)
            relation = edge_data.get('relation', 'related_to')
            bridge_triples.append((gap_start, relation, gap_end))
            logger.debug(f"Found direct bridge: ({gap_start}, {relation}, {gap_end})")

        # Strategy 2: One-hop bridge (gap_start -> bridge_entity -> gap_end)
        if len(bridge_triples) < top_k:
            if gap_start in full_kg.nodes() and gap_end in full_kg.nodes():
                start_neighbors = set(full_kg.neighbors(gap_start))
                end_neighbors = set(full_kg.neighbors(gap_end))
                bridge_entities = start_neighbors & end_neighbors

                for bridge in list(bridge_entities)[:top_k - len(bridge_triples)]:
                    # Add both edges
                    edge1_data = full_kg.get_edge_data(gap_start, bridge)
                    edge2_data = full_kg.get_edge_data(bridge, gap_end)
                    rel1 = edge1_data.get('relation', 'related_to')
                    rel2 = edge2_data.get('relation', 'related_to')
                    bridge_triples.append((gap_start, rel1, bridge))
                    bridge_triples.append((bridge, rel2, gap_end))
                    logger.debug(f"Found 1-hop bridge via: {bridge}")

        # Strategy 3: Shortest path in full KG
        if not bridge_triples and nx.has_path(full_kg, gap_start, gap_end):
            try:
                path = nx.shortest_path(full_kg, gap_start, gap_end)
                # Extract edges along the path
                for i in range(len(path) - 1):
                    edge_data = full_kg.get_edge_data(path[i], path[i+1])
                    relation = edge_data.get('relation', 'related_to')
                    bridge_triples.append((path[i], relation, path[i+1]))
                logger.debug(f"Found path bridge: {path}")
            except nx.NetworkXNoPath:
                pass

        if bridge_triples:
            self.stats['repairs_successful'] += 1
            logger.info(f"Gap repair successful: {len(bridge_triples)} triples retrieved")
        else:
            logger.warning("Gap repair failed: no bridge found in full KG")

        return bridge_triples[:top_k]

    def extract_evidence_chain(
        self,
        sub_graph: nx.Graph,
        question_entities: List[str],
        candidate_answers: List[str]
    ) -> Optional[List[Tuple[str, str, str]]]:
        """
        Extract the complete evidence chain as a list of triples.

        This is what gets passed to the LLM for generation - only the
        necessary evidence chain, not all retrieved triples.

        Returns:
            List of (head, relation, tail) forming the reasoning path,
            or None if no complete chain exists
        """
        graph_nodes = list(sub_graph.nodes())
        mapped_question_entities = [
            find_entity_in_graph(entity, graph_nodes, threshold=0.85) or entity
            for entity in question_entities
        ]
        mapped_candidate_answers = [
            find_entity_in_graph(entity, graph_nodes, threshold=0.85) or entity
            for entity in candidate_answers
        ]

        # Find the shortest path
        for q_entity in mapped_question_entities:
            if q_entity not in sub_graph:
                continue
            for answer_entity in mapped_candidate_answers:
                if answer_entity not in sub_graph:
                    continue
                try:
                    has_path = nx.has_path(sub_graph, q_entity, answer_entity)
                except nx.NodeNotFound:
                    continue
                if has_path:
                    path = nx.shortest_path(sub_graph, q_entity, answer_entity)

                    # Extract triples along the path
                    chain = []
                    for i in range(len(path) - 1):
                        edge_data = sub_graph.get_edge_data(path[i], path[i+1])
                        relation = edge_data.get('relation', 'related_to') if edge_data else 'related_to'
                        chain.append((path[i], relation, path[i+1]))

                    logger.debug(f"Extracted evidence chain: {len(chain)} triples")
                    return chain

        return None

    def check_evidence_chain_from_strings(
        self,
        triple_strings: List[str],
        question: str
    ) -> Tuple[bool, Optional[Dict], Optional[List[str]]]:
        """
        Convenience method: Check evidence chain from triple strings and question.

        This is the main entry point for pipeline integration.

        Args:
            triple_strings: List of "head -- relation --> tail" strings
            question: The original question

        Returns:
            (is_complete, gap_info, evidence_chain_strings)
            - is_complete: True if a connected path exists
            - gap_info: If incomplete, contains gap location details
            - evidence_chain_strings: If complete, the extracted evidence chain as strings

        Example:
            checker = EvidenceChainChecker()
            triples = ["A -- rel --> B", "B -- rel2 --> C"]
            is_complete, gap, chain = checker.check_evidence_chain_from_strings(triples, "What is C?")
        """
        # Build graph
        graph = build_graph_from_triple_strings(triple_strings)

        # Extract question entities
        question_entities = _filter_question_entities(extract_question_entities(question))

        # Select candidate answers
        candidate_answers = select_candidate_answers(graph, question_entities, strategy="all")
        answer_type = _infer_answer_type(question)
        self.last_answer_type = answer_type
        candidate_answers = _filter_candidates_by_answer_type(candidate_answers, answer_type)
        min_edges = _minimum_chain_edges(question)
        candidate_answers = _filter_candidates_by_min_distance(
            graph,
            question_entities,
            candidate_answers,
            min_edges,
        )

        # Check completeness
        is_complete, gap_info = self.check_evidence_chain(
            graph,
            question_entities,
            candidate_answers
        )

        # If complete, extract the evidence chain
        evidence_chain_strings = None
        if is_complete:
            chain_triples = self.extract_evidence_chain(graph, question_entities, candidate_answers)
            if chain_triples:
                evidence_chain_strings = triples_to_strings(chain_triples)
                if len(evidence_chain_strings) < min_edges:
                    return False, {
                        'gap_type': 'short_chain',
                        'chain_length': len(evidence_chain_strings),
                        'min_chain_length': min_edges,
                        'explanation': (
                            f"Evidence path has {len(evidence_chain_strings)} edge(s), "
                            f"below the required {min_edges} for this question type"
                        )
                    }, evidence_chain_strings

        return is_complete, gap_info, evidence_chain_strings


    def get_statistics(self) -> Dict:
        """Return checker statistics for analysis."""
        stats = self.stats.copy()
        if stats['total_checks'] > 0:
            stats['chain_completeness_rate'] = stats['complete_chains'] / stats['total_checks']
        if stats['repairs_attempted'] > 0:
            stats['repair_success_rate'] = stats['repairs_successful'] / stats['repairs_attempted']
        return stats


def build_graph_from_triples(triples: List[Tuple[str, str, str]]) -> nx.Graph:
    """
    Build a NetworkX graph from a list of triples.

    Args:
        triples: List of (head, relation, tail)

    Returns:
        NetworkX Graph with nodes and edges
    """
    G = nx.Graph()
    for head, relation, tail in triples:
        G.add_edge(head, tail, relation=relation)
    return G


# ============================================================
# Unit Tests (to be run with pytest)
# ============================================================

def test_complete_chain():
    """Test case: Complete evidence chain exists."""
    checker = EvidenceChainChecker()

    # Simple chain: A -> B -> C
    triples = [
        ("Inception", "directed_by", "Christopher Nolan"),
        ("Christopher Nolan", "nationality", "British"),
        ("British", "country", "UK")
    ]
    graph = build_graph_from_triples(triples)

    is_complete, gap_info = checker.check_evidence_chain(
        graph,
        question_entities=["Inception"],
        candidate_answers=["UK"]
    )

    assert is_complete == True
    assert gap_info is None
    print("✓ Test 1 passed: Complete chain detected")


def test_incomplete_chain():
    """Test case: Evidence chain has a gap."""
    checker = EvidenceChainChecker()

    # Disconnected: A -> B, D -> E (missing B -> D)
    triples = [
        ("Inception", "directed_by", "Christopher Nolan"),
        ("UK", "capital", "London")
    ]
    graph = build_graph_from_triples(triples)

    is_complete, gap_info = checker.check_evidence_chain(
        graph,
        question_entities=["Inception"],
        candidate_answers=["UK"]
    )

    assert is_complete == False
    assert gap_info is not None
    assert gap_info['gap_type'] == 'disconnected'
    print("✓ Test 2 passed: Gap detected")


def test_targeted_repair():
    """Test case: Targeted gap repair finds bridge triple."""
    checker = EvidenceChainChecker()

    # Retrieved subgraph (incomplete)
    retrieved_triples = [
        ("Inception", "directed_by", "Christopher Nolan"),
        ("UK", "capital", "London")
    ]
    sub_graph = build_graph_from_triples(retrieved_triples)

    # Full KG (has the missing link)
    full_triples = retrieved_triples + [
        ("Christopher Nolan", "nationality", "British"),
        ("British", "is_a", "UK")
    ]
    full_kg = build_graph_from_triples(full_triples)

    # Check for gap
    is_complete, gap_info = checker.check_evidence_chain(
        sub_graph,
        ["Inception"],
        ["UK"]
    )
    assert is_complete == False

    # Repair
    bridge_triples = checker.targeted_gap_repair(full_kg, gap_info)
    assert len(bridge_triples) > 0
    print(f"✓ Test 3 passed: Repair found {len(bridge_triples)} bridge triples")


def test_extract_evidence_chain():
    """Test case: Extract complete evidence chain."""
    checker = EvidenceChainChecker()

    triples = [
        ("Inception", "directed_by", "Christopher Nolan"),
        ("Christopher Nolan", "nationality", "British"),
        ("British", "country", "UK"),
        # Extra noise
        ("Dark Knight", "directed_by", "Christopher Nolan"),
        ("UK", "capital", "London")
    ]
    graph = build_graph_from_triples(triples)

    chain = checker.extract_evidence_chain(
        graph,
        ["Inception"],
        ["UK"]
    )

    assert chain is not None
    assert len(chain) == 3  # Only the path, not the noise
    print(f"✓ Test 4 passed: Extracted chain with {len(chain)} triples")


def test_empty_inputs():
    """Test case: Handle empty inputs gracefully."""
    checker = EvidenceChainChecker()
    graph = nx.Graph()

    is_complete, gap_info = checker.check_evidence_chain(
        graph,
        [],
        ["UK"]
    )

    assert is_complete == False
    assert gap_info['gap_type'] == 'empty_input'
    print("✓ Test 5 passed: Empty inputs handled")


def test_check_from_strings():
    """Test the convenience method for pipeline integration."""
    checker = EvidenceChainChecker()

    # Complete chain
    triples = [
        "Inception -- directed_by --> Christopher Nolan",
        "Christopher Nolan -- nationality --> British",
        "British -- country --> UK"
    ]
    question = "What country is the director of Inception from?"

    is_complete, gap_info, chain = checker.check_evidence_chain_from_strings(triples, question)

    assert is_complete == True
    assert gap_info is None
    assert chain is not None
    assert len(chain) > 0
    print(f"✓ Test 6 passed: check_from_strings works, chain: {chain}")


if __name__ == "__main__":
    print("Running Evidence Chain Checker unit tests...\n")
    test_complete_chain()
    test_incomplete_chain()
    test_targeted_repair()
    test_extract_evidence_chain()
    test_empty_inputs()
    test_check_from_strings()
    print("\n✅ All tests passed!")
