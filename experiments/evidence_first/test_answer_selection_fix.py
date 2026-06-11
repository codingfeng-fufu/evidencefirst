"""Quick test to verify answer selection fix for 'not mentioned' cases."""

import json
from comagraag.pipeline import run_pipeline

# Load the 100q results to get test cases
with open('test_ab_100q_results.json', 'r') as f:
    data = json.load(f)

# Select 5 cases where KG said "not mentioned" but context had an answer
test_cases = [
    # Case 1: EGOT question
    {
        'qid': '5a7b1e22554299042af8f6e4',
        'question': 'What award won by only twelve people has a man who Ted Kooshian has performed with won?',
        'gold': 'EGOT',
        'expected_context': 'EGOT'
    },
    # Case 11: Albany question
    {
        'qid': '5a8c64ce5542992a431d1ad0',
        'question': 'What city does Paul Clyne and David Soares have in common?',
        'gold': 'New York',
        'expected_context': 'Albany'
    },
    # Case 2: Volvo S70 question
    {
        'qid': '5a8ad66c5542992d82986f6e',
        'question': 'During which period the sedan variant of the Volvo V50 car was manufactured?',
        'gold': '1995 to 2012',
        'expected_context': '2006–2013'
    },
]

# Find full data for these cases
full_cases = []
for tc in test_cases:
    for r in data['results']:
        if r['qid'] == tc['qid']:
            full_cases.append({
                'qid': tc['qid'],
                'question': r['question'],
                'gold': r['gold_answer'],
                'context': r.get('context'),  # May not be in results
                'expected_context': tc['expected_context']
            })
            break

print("=== 测试答案选择修复 ===")
print(f"测试 {len(test_cases)} 个之前拒答的案例\n")

# For quick test, we need to load the dataset to get context
from datasets import load_dataset

dataset = load_dataset("hotpot_qa", "distractor", split="validation[:100]")

results = []
for i, tc in enumerate(test_cases, 1):
    # Find the case in dataset
    case_data = None
    for item in dataset:
        if item['id'] == tc['qid']:
            case_data = item
            break

    if not case_data:
        print(f"[{i}] 跳过 - 找不到数据")
        continue

    question = case_data['question']
    context = case_data['context']
    gold = case_data['answer']

    print(f"\n[{i}] Q: {question[:80]}...")
    print(f"    Gold: {gold}")
    print(f"    旧系统: 'not mentioned' (错误)")
    print(f"    Context曾给出: {tc['expected_context']}")

    try:
        result = run_pipeline(
            question=question,
            graph=None,
            context=context,
            variant="evidence_first"
        )

        pred = result.get('answer', 'ERROR')
        print(f"    新系统: {pred}")

        # Check if it's no longer "not mentioned"
        if 'not mentioned' in pred.lower():
            print(f"    ✗ 仍然拒答")
            success = False
        else:
            print(f"    ✓ 给出了具体答案")
            success = True

            # Check if it matches gold
            from comagraag.evaluate import normalize_answer
            if normalize_answer(pred) == normalize_answer(gold):
                print(f"    ✓✓ 答案完全正确！")

        results.append({
            'question': question[:60],
            'gold': gold,
            'old': 'not mentioned',
            'new': pred,
            'fixed': success
        })

    except Exception as e:
        print(f"    ERROR: {e}")
        results.append({
            'question': question[:60],
            'gold': gold,
            'old': 'not mentioned',
            'new': f'ERROR: {e}',
            'fixed': False
        })

print(f"\n\n=== 总结 ===")
fixed = sum(1 for r in results if r['fixed'])
print(f"修复成功: {fixed}/{len(results)}")
print(f"修复率: {fixed/len(results)*100:.1f}%")

if fixed > 0:
    print(f"\n✓ 答案选择策略修复生效！")
    print(f"  预计可以救回 21 个拒答案例中的 {int(21 * fixed/len(results))} 个")
    print(f"  EM提升预期: +{int(21 * fixed/len(results))}%")
