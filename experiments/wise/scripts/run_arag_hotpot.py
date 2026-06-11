import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARAG_ROOT = ROOT / "external_repos" / "arag"
ARAG_SRC = ARAG_ROOT / "src"

sys.path.insert(0, str(ARAG_SRC))


EXACT_MATCH_SYSTEM_PROMPT = """You are a question-answering assistant with access to a document corpus through available tools.

Use the available retrieval tools iteratively to find the evidence needed to answer the user's question. For multi-hop questions, break the problem into factual steps internally and search/read until the answer is supported by retrieved text.

When you have enough evidence, stop calling tools and output only the final answer text expected by an exact-match QA evaluator.

Final-answer rules:
- Output only the concise answer string.
- Do not include citations, markdown, explanations, or "Answer:" prefixes.
- Preserve names, dates, numbers, and yes/no answers exactly as supported by the evidence.
- If the evidence supports a descriptive answer rather than a named entity, output the shortest descriptive phrase or sentence that directly answers the question.
"""

ANSWER_ONLY_SUFFIX = (
    "\n\nReturn only the concise final answer text. Do not include reasoning, "
    "citations, markdown, or an 'Answer:' prefix."
)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _load_batch_runner():
    runner_path = ARAG_ROOT / "scripts" / "batch_runner.py"
    spec = importlib.util.spec_from_file_location("arag_batch_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _hotpot_runner_class(batch_runner_module):
    class HotpotBatchRunner(batch_runner_module.BatchRunner):
        def _create_answer_cleaner(self):
            llm_config = self.config.get("llm", {})
            return batch_runner_module.LLMClient(
                model=llm_config.get("model") or os.getenv("ARAG_MODEL", "gpt-4o-mini"),
                api_key=llm_config.get("api_key") or os.getenv("ARAG_API_KEY"),
                base_url=llm_config.get("base_url") or os.getenv("ARAG_BASE_URL", "https://api.openai.com/v1"),
                reasoning_effort=llm_config.get("reasoning_effort"),
                max_tokens=128,
            )

        def _extract_final_answer(self, question, raw_answer):
            prompt = (
                "Extract the concise final answer from the model response for exact-match QA scoring.\n"
                "Do not solve the question independently and do not add facts that are not stated or directly implied "
                "by the response.\n"
                "Return only the answer text, with no explanation, markdown, citations, or prefix.\n\n"
                f"Question: {question}\n\n"
                f"Model response:\n{raw_answer}"
            )
            response = self._answer_cleaner.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                temperature=0.0,
                max_tokens=128,
            )
            return response["message"].get("content", "").strip(), response["cost"]

        def _process_one(self, item, agent):
            original_question = item.get("question", "")
            prompted_item = dict(item)
            prompted_item["question"] = f"{original_question}{ANSWER_ONLY_SUFFIX}"
            result = None
            for attempt in range(3):
                result = super()._process_one(prompted_item, agent)
                answer = str(result.get("pred_answer", ""))
                if not result.get("error") and not answer.startswith("Error:"):
                    break
                if attempt < 2:
                    time.sleep(2 ** attempt)
            assert result is not None
            result["question"] = original_question
            raw_answer = str(result.get("pred_answer", ""))
            if raw_answer.startswith("Error:") and not result.get("error"):
                result["error"] = raw_answer
            if raw_answer and not raw_answer.startswith("Error:"):
                try:
                    cleaned_answer, cleaner_cost = self._extract_final_answer(
                        original_question, raw_answer
                    )
                    result["raw_pred_answer"] = raw_answer
                    result["pred_answer"] = cleaned_answer
                    result["answer_postprocess_cost"] = cleaner_cost
                    result["total_cost"] = float(result.get("total_cost", 0) or 0) + cleaner_cost
                except Exception as exc:
                    result["answer_postprocess_error"] = str(exc)
            return result

    return HotpotBatchRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run official A-RAG BatchRunner with an exact-match QA-compatible final-answer prompt."
    )
    parser.add_argument("--config", required=True, help="A-RAG YAML config path")
    parser.add_argument("--questions", required=True, help="Questions JSON path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--workers", type=int, default=1, help="Number of runner workers")
    parser.add_argument("--env-file", type=Path, default=ROOT / "comagraag" / ".env")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _load_env_file(args.env_file)
    os.environ.setdefault("ARAG_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    os.environ.setdefault(
        "ARAG_BASE_URL",
        os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    os.environ.setdefault("ARAG_MODEL", os.environ.get("LLM_MODEL", "qwen-plus"))

    if not os.environ.get("ARAG_API_KEY"):
        raise SystemExit("ARAG_API_KEY or OPENAI_API_KEY is required.")

    batch_runner = _load_batch_runner()
    config = batch_runner.Config.from_yaml(args.config)
    runner_cls = _hotpot_runner_class(batch_runner)
    runner = runner_cls(
        config=config,
        questions_file=args.questions,
        output_dir=args.output,
        limit=args.limit,
        num_workers=args.workers,
        verbose=args.verbose,
    )
    runner._system_prompt = EXACT_MATCH_SYSTEM_PROMPT
    runner._answer_cleaner = runner._create_answer_cleaner()
    runner.run()


if __name__ == "__main__":
    main()
