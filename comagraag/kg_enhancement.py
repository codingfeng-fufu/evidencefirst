"""
KG Enhancement Module: A+B Solution
A+: Dynamic entity supplementation - extract triples for missing entities
B: Gap repair - find bridging triples between disconnected entities
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


def llm_extract_triples_for_entity(
    entity: str,
    context: str,
    model: str = "gpt-4o-mini"
) -> List[Tuple[str, str, str]]:
    """
    A+ Solution: Extract triples related to a specific entity from context passages.

    Args:
        entity: The entity to extract triples for
        context: The context passages (e.g., retrieved passages from GRA)
        model: Model to use for extraction

    Returns:
        List of triples: [(subject, relation, object), ...]
    """
    if not entity or not context:
        return []

    prompt = f"""从以下段落中抽取所有与实体"{entity}"相关的事实三元组。

要求：
1. 三元组格式：(主体, 关系, 客体)
2. 只抽取直接相关的事实
3. 每行一个三元组
4. 确保实体名称准确

段落：
{context}

请直接输出三元组列表，每行格式：(主体, 关系, 客体)
"""

    try:
        # Import llm_call from agents
        from agents import llm_call

        response = llm_call(prompt, max_tokens=500)

        # Parse triples from response
        triples = []
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Match pattern: (subject, relation, object)
            match = re.match(r'\((.+?),\s*(.+?),\s*(.+?)\)', line)
            if match:
                subject = match.group(1).strip()
                relation = match.group(2).strip()
                obj = match.group(3).strip()
                triples.append((subject, relation, obj))

        logger.info(f"A+ extracted {len(triples)} triples for entity '{entity}'")
        return triples

    except Exception as e:
        logger.error(f"A+ extraction failed for entity '{entity}': {e}")
        return []


def llm_extract_bridging_triples(
    entity_a: str,
    entity_b: str,
    context: str,
    model: str = "gpt-4o-mini"
) -> List[Tuple[str, str, str]]:
    """
    B Solution: Find bridging triples that connect two disconnected entities.

    Args:
        entity_a: First entity
        entity_b: Second entity
        context: The context passages
        model: Model to use for extraction

    Returns:
        List of bridging triples: [(subject, relation, object), ...]
    """
    if not entity_a or not entity_b or not context:
        return []

    prompt = f"""从以下段落中抽取能够连接"{entity_a}"和"{entity_b}"这两个实体的桥接三元组。

目标：找到一条或多条三元组路径，使得"{entity_a}"和"{entity_b}"可以通过知识图谱连接起来。

要求：
1. 三元组格式：(主体, 关系, 客体)
2. 三元组应该形成连接路径
3. 每行一个三元组
4. 确保实体名称准确

段落：
{context}

请直接输出三元组列表，每行格式：(主体, 关系, 客体)
"""

    try:
        # Import llm_call from agents
        from agents import llm_call

        response = llm_call(prompt, max_tokens=500)

        # Parse triples from response
        triples = []
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Match pattern: (subject, relation, object)
            match = re.match(r'\((.+?),\s*(.+?),\s*(.+?)\)', line)
            if match:
                subject = match.group(1).strip()
                relation = match.group(2).strip()
                obj = match.group(3).strip()
                triples.append((subject, relation, obj))

        logger.info(f"B extracted {len(triples)} bridging triples for '{entity_a}' <-> '{entity_b}'")
        return triples

    except Exception as e:
        logger.error(f"B extraction failed for '{entity_a}' <-> '{entity_b}': {e}")
        return []


def add_triples_to_graph(graph: Any, triples: List[Tuple[str, str, str]]) -> int:
    """
    Add triples to the knowledge graph.

    Args:
        graph: NetworkX graph object
        triples: List of triples to add

    Returns:
        Number of triples successfully added
    """
    if not triples:
        return 0

    added = 0
    for subject, relation, obj in triples:
        try:
            # Add edge with relation as attribute
            graph.add_edge(subject, obj, relation=relation)
            added += 1
        except Exception as e:
            logger.warning(f"Failed to add triple ({subject}, {relation}, {obj}): {e}")

    return added


def enhance_kg_with_missing_entities(
    missing_entities: List[str],
    context: str,
    sub_graph: Any,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    A+ Solution: Enhance KG by extracting triples for missing entities.

    Args:
        missing_entities: List of missing entities
        context: Context passages
        sub_graph: The sub-graph to enhance
        model: Model to use

    Returns:
        Enhancement statistics
    """
    stats = {
        'entities_processed': 0,
        'triples_extracted': 0,
        'triples_added': 0,
        'llm_calls': 0
    }

    for entity in missing_entities:
        # Extract triples for this entity
        triples = llm_extract_triples_for_entity(entity, context, model)
        stats['llm_calls'] += 1
        stats['entities_processed'] += 1
        stats['triples_extracted'] += len(triples)

        # Add triples to graph
        added = add_triples_to_graph(sub_graph, triples)
        stats['triples_added'] += added

    logger.info(f"A+ enhancement: {stats}")
    return stats


def repair_disconnected_entities(
    disconnected_pairs: List[Tuple[str, str]],
    context: str,
    sub_graph: Any,
    model: str = "gpt-4o-mini",
    max_pairs: int = 5
) -> Dict[str, Any]:
    """
    B Solution: Repair disconnected entity pairs by finding bridging triples.

    Args:
        disconnected_pairs: List of disconnected entity pairs
        context: Context passages
        sub_graph: The sub-graph to repair
        model: Model to use
        max_pairs: Maximum number of pairs to repair (cost control)

    Returns:
        Repair statistics
    """
    stats = {
        'pairs_processed': 0,
        'triples_extracted': 0,
        'triples_added': 0,
        'llm_calls': 0
    }

    # Limit the number of pairs to process
    pairs_to_process = disconnected_pairs[:max_pairs]

    for entity_a, entity_b in pairs_to_process:
        # Extract bridging triples
        triples = llm_extract_bridging_triples(entity_a, entity_b, context, model)
        stats['llm_calls'] += 1
        stats['pairs_processed'] += 1
        stats['triples_extracted'] += len(triples)

        # Add triples to graph
        added = add_triples_to_graph(sub_graph, triples)
        stats['triples_added'] += added

    logger.info(f"B repair: {stats}")
    return stats
