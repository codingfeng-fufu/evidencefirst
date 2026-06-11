# Blind Multi-Agent Gap-Label Adjudication Packet

Task: judge whether the system structural gap label is semantically plausible from the visible evidence packet.

Labels:
- match: the system gap label is a plausible explanation of the visible evidence state for this question/prediction.
- mismatch: the visible evidence strongly contradicts the system gap label.
- ambiguous: the packet is insufficient or mixed, so a reviewer should not treat the label as validated or invalidated.

Important boundaries:
- Do not use external web search.
- Do not infer from gold answers, EM/F1, baseline outputs, or prior review notes; they are intentionally absent.
- Treat complete as: visible evidence appears sufficient for the predicted answer under the question, not as formal proof.
- Treat missing_entities/disconnected/short_chain as operational routing labels; decide if the visible evidence and structural fields make that routing label plausible.

Required output: JSON array, one object per case, with fields:
case_id, judgment (match|mismatch|ambiguous), confidence (low|medium|high), rationale (one short sentence).
