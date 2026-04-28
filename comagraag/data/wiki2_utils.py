def normalize_context(context) -> list:
    """
    把 2WikiMultiHopQA 的 context 统一为 HotpotQA 格式：
    [[title, [sent1, sent2, ...]], ...]
    """
    if isinstance(context, dict):
        return [[title, sents] for title, sents in context.items()]

    elif isinstance(context, list):
        if not context:
            return []
        first = context[0]

        if isinstance(first, (list, tuple)) and len(first) == 2:
            if isinstance(first[1], list):
                return [[item[0], item[1]] for item in context]
            return []

        if isinstance(first, str):
            return [["document", context]]

    return []
