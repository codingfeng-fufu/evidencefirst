"""
Generate quality analysis figures for WISE 2026 paper
Creates publication-quality figures for:
1. Convergence-correctness correlation (bar chart)
2. VA score distribution (density curves)
3. Case study surgical repair flow (visualization)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# Load data
with open('results/quality_analysis_full.json', 'r') as f:
    quality_data = json.load(f)

with open('results/case_studies.json', 'r') as f:
    case_studies = json.load(f)

# Create output directory
output_dir = Path('results/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Figure 1: Convergence-Correctness Correlation (Bar Chart)
# ============================================================
print("Generating Figure 1: Convergence-Correctness Correlation...")

fig, ax = plt.subplots(figsize=(6, 4))

categories = ['Converged\n(VA ≥ 0.5)', 'Not Converged\n(VA < 0.5)']
em_values = [
    quality_data['convergence']['converged_em'] * 100,
    quality_data['convergence']['not_converged_em'] * 100
]
counts = [
    quality_data['convergence']['converged_count'],
    quality_data['convergence']['not_converged_count']
]

colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(categories, em_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for i, (bar, val, count) in enumerate(zip(bars, em_values, counts)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val:.1f}%\n(n={count})',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add gap annotation
gap = quality_data['convergence']['em_gap'] * 100
ax.annotate('', xy=(1, em_values[1]), xytext=(1, em_values[0]),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(1.15, (em_values[0] + em_values[1])/2, f'Gap:\n{gap:.1f}%',
        va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

ax.set_ylabel('Exact Match (%)', fontweight='bold')
ax.set_xlabel('VA Convergence Status', fontweight='bold')
ax.set_title('Convergence-Correctness Correlation\n(1000 HotpotQA Questions)',
             fontweight='bold', pad=15)
ax.set_ylim(0, 75)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / 'convergence_correctness_correlation.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'convergence_correctness_correlation.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved to {output_dir}/convergence_correctness_correlation.pdf")
plt.close()

# ============================================================
# Figure 2: VA Score Distribution (Density Curves)
# ============================================================
print("\nGenerating Figure 2: VA Score Distribution...")

# Load full cache to get individual VA scores
print("  Loading cache data for score distributions...")
with open('results/cache_wise_hotpot1000_comagraag_full_fresh_combined1000_full.json', 'r') as f:
    cache_data = json.load(f)

# Load gold answers
print("  Loading gold answers...")
with open('data/hotpotqa_1000_test.json', 'r') as f:
    gold_data = json.load(f)

# Create gold answer lookup
gold_answers = {}
for item in gold_data:
    gold_answers[item['id']] = item['answer']

# Normalize answer function (from HotpotQA evaluation)
def normalize_answer(s):
    import string
    import re
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Extract VA scores for correct vs wrong answers
correct_scores = []
wrong_scores = []

for qid, item in cache_data.items():
    if 'score' in item and qid in gold_answers:
        score = item['score']
        pred = normalize_answer(item.get('answer', ''))
        gold = normalize_answer(gold_answers[qid])
        is_correct = (pred == gold)

        if is_correct:
            correct_scores.append(score)
        else:
            wrong_scores.append(score)

print(f"  Found {len(correct_scores)} correct, {len(wrong_scores)} wrong answers")

fig, ax = plt.subplots(figsize=(7, 4.5))

# Plot density curves
from scipy import stats
correct_kde = stats.gaussian_kde(correct_scores)
wrong_kde = stats.gaussian_kde(wrong_scores)

x_range = np.linspace(0, 1, 200)
correct_density = correct_kde(x_range)
wrong_density = wrong_kde(x_range)

ax.plot(x_range, correct_density, linewidth=2.5, color='#2ecc71',
        label=f'Correct Answers (n={len(correct_scores)})', alpha=0.9)
ax.fill_between(x_range, correct_density, alpha=0.2, color='#2ecc71')

ax.plot(x_range, wrong_density, linewidth=2.5, color='#e74c3c',
        label=f'Wrong Answers (n={len(wrong_scores)})', alpha=0.9)
ax.fill_between(x_range, wrong_density, alpha=0.2, color='#e74c3c')

# Add vertical lines for mean scores
correct_mean = quality_data['va_scores']['avg_score_correct']
wrong_mean = quality_data['va_scores']['avg_score_wrong']

ax.axvline(correct_mean, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Mean (Correct): {correct_mean:.3f}')
ax.axvline(wrong_mean, color='#c0392b', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Mean (Wrong): {wrong_mean:.3f}')

# Add threshold line
ax.axvline(0.5, color='black', linestyle=':', linewidth=2, alpha=0.5,
           label='VA Threshold (θ=0.5)')

# Add difference annotation
diff = quality_data['va_scores']['score_difference']
ax.text(0.75, max(correct_density.max(), wrong_density.max()) * 0.85,
        f'Score Difference:\n{diff:.3f}',
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.4))

ax.set_xlabel('VA Quality Score', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('VA Score Distribution: Correct vs. Wrong Answers\n(1000 HotpotQA Questions)',
             fontweight='bold', pad=15)
ax.legend(loc='upper left', framealpha=0.95)
ax.grid(alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(output_dir / 'va_score_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'va_score_distribution.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved to {output_dir}/va_score_distribution.pdf")
plt.close()

# ============================================================
# Figure 3: Iteration Analysis (Stacked Bar)
# ============================================================
print("\nGenerating Figure 3: Iteration Analysis...")

fig, ax = plt.subplots(figsize=(6, 4))

iter1_count = quality_data['iterations']['iter_1_count']
iter2_count = quality_data['iterations']['iter_2_count']
total = iter1_count + iter2_count

iter1_pct = (iter1_count / total) * 100
iter2_pct = (iter2_count / total) * 100

# Create stacked bar
categories = ['All Questions']
bars1 = ax.barh(categories, iter1_pct, color='#3498db',
                edgecolor='black', linewidth=1.2, label='Converged at Iteration 1')
bars2 = ax.barh(categories, iter2_pct, left=iter1_pct, color='#e67e22',
                edgecolor='black', linewidth=1.2, label='Required Iteration 2')

# Add percentage labels
ax.text(iter1_pct/2, 0, f'{iter1_pct:.1f}%\n(n={iter1_count})',
        ha='center', va='center', fontsize=11, fontweight='bold', color='white')
ax.text(iter1_pct + iter2_pct/2, 0, f'{iter2_pct:.1f}%\n(n={iter2_count})',
        ha='center', va='center', fontsize=11, fontweight='bold', color='white')

ax.set_xlabel('Percentage of Questions (%)', fontweight='bold')
ax.set_title('Iterative Refinement Effectiveness\n(1000 HotpotQA Questions)',
             fontweight='bold', pad=15)
ax.set_xlim(0, 100)
ax.legend(loc='lower right', framealpha=0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add average iterations annotation
avg_iter = quality_data['iterations']['average']
ax.text(50, 0.3, f'Average Iterations: {avg_iter:.2f}',
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / 'iteration_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'iteration_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved to {output_dir}/iteration_analysis.pdf")
plt.close()

# ============================================================
# Figure 4: Case Study Surgical Repair (Flow Diagram)
# ============================================================
print("\nGenerating Figure 4: Case Study Surgical Repair Flow...")

# Use the best case study (highest improvement)
best_case = case_studies[0]

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Surgical Sub-Query Repair: Case Study Example',
        ha='center', fontsize=14, fontweight='bold')

# Question box
question_text = best_case['question'][:100] + '...'
ax.add_patch(plt.Rectangle((0.5, 7.8), 9, 1.2,
                            facecolor='lightblue', edgecolor='black', linewidth=2))
ax.text(5, 8.4, 'Question:', ha='center', fontsize=10, fontweight='bold')
ax.text(5, 8.1, question_text, ha='center', fontsize=8, wrap=True)

# Iteration 1
ax.add_patch(plt.Rectangle((0.5, 5.8), 4, 1.5,
                            facecolor='#ffcccc', edgecolor='red', linewidth=2))
ax.text(2.5, 7, 'Iteration 1', ha='center', fontsize=11, fontweight='bold')
ax.text(2.5, 6.7, f'VA Score: {best_case["iter1_score"]:.2f}',
        ha='center', fontsize=9, color='red', fontweight='bold')
ax.text(2.5, 6.4, 'Sub-queries:', ha='center', fontsize=8, style='italic')
sq1 = best_case['iter1_subqueries'][0][:40] + '...'
ax.text(2.5, 6.1, f'1. {sq1}', ha='center', fontsize=7)

# VA Feedback (middle)
ax.add_patch(plt.Rectangle((5.5, 5.8), 4, 1.5,
                            facecolor='#fff4cc', edgecolor='orange', linewidth=2))
ax.text(7.5, 7, 'VA Diagnostic Feedback', ha='center', fontsize=11, fontweight='bold')
feedback_short = best_case['iter1_feedback'][:80] + '...'
ax.text(7.5, 6.4, feedback_short, ha='center', fontsize=7, wrap=True)
ax.text(7.5, 6.0, '⚠️ Entity linking failed', ha='center', fontsize=8,
        color='red', fontweight='bold')

# Arrow 1 -> Feedback
ax.annotate('', xy=(5.3, 6.5), xytext=(4.7, 6.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))
ax.text(5, 6.8, 'Fails', ha='center', fontsize=8, color='red', fontweight='bold')

# Arrow Feedback -> 2
ax.annotate('', xy=(5.3, 4.5), xytext=(7.5, 5.6),
            arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
ax.text(6.2, 5.2, 'Surgical\nRepair', ha='center', fontsize=8,
        color='orange', fontweight='bold')

# Iteration 2
ax.add_patch(plt.Rectangle((0.5, 3.2), 4, 1.5,
                            facecolor='#ccffcc', edgecolor='green', linewidth=2))
ax.text(2.5, 4.4, 'Iteration 2', ha='center', fontsize=11, fontweight='bold')
ax.text(2.5, 4.1, f'VA Score: {best_case["iter2_score"]:.2f} ✓',
        ha='center', fontsize=9, color='green', fontweight='bold')
ax.text(2.5, 3.8, 'Revised sub-queries:', ha='center', fontsize=8, style='italic')
sq2 = best_case['iter2_subqueries'][0][:40] + '...'
ax.text(2.5, 3.5, f'1. {sq2}', ha='center', fontsize=7)

# Final answer
ax.add_patch(plt.Rectangle((5.5, 3.2), 4, 1.5,
                            facecolor='#ccffdd', edgecolor='darkgreen', linewidth=2))
ax.text(7.5, 4.4, 'Final Answer', ha='center', fontsize=11, fontweight='bold')
ax.text(7.5, 4.0, f'"{best_case["final_answer"]}"',
        ha='center', fontsize=10, color='darkgreen', fontweight='bold')
ax.text(7.5, 3.6, '✅ Matches Gold Answer',
        ha='center', fontsize=9, color='green', fontweight='bold')

# Arrow 2 -> Answer
ax.annotate('', xy=(5.3, 4.0), xytext=(4.7, 4.0),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax.text(5, 4.3, 'Succeeds', ha='center', fontsize=8, color='green', fontweight='bold')

# Key insight box
ax.add_patch(plt.Rectangle((0.5, 0.5), 9, 1.2,
                            facecolor='lightyellow', edgecolor='black', linewidth=2, linestyle='--'))
ax.text(5, 1.4, '🔑 Key Insight: Targeted Revision',
        ha='center', fontsize=11, fontweight='bold')
ax.text(5, 0.9, f'Score improvement: +{best_case["score_improvement"]:.2f} | ' +
        'Only failing sub-query revised, successful ones preserved',
        ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'case_study_surgical_repair.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'case_study_surgical_repair.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved to {output_dir}/case_study_surgical_repair.pdf")
plt.close()

# ============================================================
# Summary Statistics Table
# ============================================================
print("\n" + "="*60)
print("QUALITY ANALYSIS SUMMARY")
print("="*60)
print(f"\n📊 Overall Performance:")
print(f"  • Total questions: {quality_data['overall']['total_questions']}")
print(f"  • Correct: {quality_data['overall']['correct']}")
print(f"  • Exact Match: {quality_data['overall']['em']:.3f} ({quality_data['overall']['em']*100:.1f}%)")

print(f"\n✅ Convergence Analysis:")
print(f"  • Converged: {quality_data['convergence']['converged_count']} ({quality_data['convergence']['converged_rate']*100:.1f}%)")
print(f"  • Converged EM: {quality_data['convergence']['converged_em']:.3f} ({quality_data['convergence']['converged_em']*100:.1f}%)")
print(f"  • Not Converged EM: {quality_data['convergence']['not_converged_em']:.3f} ({quality_data['convergence']['not_converged_em']*100:.1f}%)")
print(f"  • EM Gap: {quality_data['convergence']['em_gap']:.3f} ({quality_data['convergence']['em_gap']*100:.1f}%) ⭐")

print(f"\n📈 VA Score Analysis:")
print(f"  • Avg score (correct): {quality_data['va_scores']['avg_score_correct']:.3f}")
print(f"  • Avg score (wrong): {quality_data['va_scores']['avg_score_wrong']:.3f}")
print(f"  • Score difference: {quality_data['va_scores']['score_difference']:.3f} ⭐")

print(f"\n🔄 Iteration Analysis:")
print(f"  • Average iterations: {quality_data['iterations']['average']:.2f}")
print(f"  • Converged at iter 1: {quality_data['iterations']['iter_1_count']} ({quality_data['iterations']['iter_1_count']/10:.1f}%)")
print(f"  • Required iter 2: {quality_data['iterations']['iter_2_count']} ({quality_data['iterations']['iter_2_count']/10:.1f}%) ⭐")

print(f"\n💡 Top Case Study:")
print(f"  • Question ID: {best_case['qid']}")
print(f"  • Score improvement: +{best_case['score_improvement']:.2f}")
print(f"  • Iter 1 → Iter 2: {best_case['iter1_score']:.2f} → {best_case['iter2_score']:.2f}")

print("\n" + "="*60)
print("✅ All figures generated successfully!")
print("="*60)
print(f"\nOutput directory: {output_dir.absolute()}")
print("\nGenerated files:")
print("  1. convergence_correctness_correlation.pdf/.png")
print("  2. va_score_distribution.pdf/.png")
print("  3. iteration_analysis.pdf/.png")
print("  4. case_study_surgical_repair.pdf/.png")
print("\nThese figures are ready for inclusion in the WISE 2026 paper.")
