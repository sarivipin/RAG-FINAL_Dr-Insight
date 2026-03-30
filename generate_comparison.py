import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# LOAD DATA
# =========================================================
with open("ragas_results.csv") as f:
    ollama_lines = f.read().strip().split("\n")

with open("ragas_results_groq.csv") as f:
    groq_lines = f.read().strip().split("\n")

with open("ragas_details.json") as f:
    ollama_details = json.load(f)

with open("ragas_details_groq.json") as f:
    groq_details = json.load(f)

# Parse per-question scores
questions_short = [
    "Bacterial\nVaginosis",
    "Varicose\nVeins",
    "Melanoma\nPrevention",
    "Type 1\nDiabetes",
    "MRSA",
]

ollama_faith = [0.80, 0.80, 0.33, 0.50, 0.40]
ollama_recall = [0.40, 0.40, 0.40, 0.33, 0.40]
ollama_bleu = [0.911, 0.459, 0.785, 0.096, 0.342]

groq_faith = [1.0, 1.0, 1.0, 1.0, 1.0]
groq_recall = [1.0, 1.0, 1.0, 1.0, 1.0]
groq_bleu = [0.868, 0.361, 0.969, 0.200, 0.786]

# Averages
ollama_avg = [np.mean(ollama_faith), np.mean(ollama_recall), np.mean(ollama_bleu)]
groq_avg = [np.mean(groq_faith), np.mean(groq_recall), np.mean(groq_bleu)]

colors_ollama = "#4A90D9"
colors_groq = "#E8833A"

# =========================================================
# CHART 1: Overall Averages Bar Chart
# =========================================================
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ["Faithfulness", "Context Recall", "BLEU Score"]
x = np.arange(len(metrics))
width = 0.32

bars1 = ax.bar(x - width/2, [v * 100 for v in ollama_avg], width, label="Ollama (llama3.2 - 3B)", color=colors_ollama, edgecolor="white")
bars2 = ax.bar(x + width/2, [v * 100 for v in groq_avg], width, label="Groq (llama-3.3-70b)", color=colors_groq, edgecolor="white")

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Overall Evaluation: Ollama vs Groq", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 115)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("chart_overall_comparison.png", dpi=150)
plt.close()
print("Saved chart_overall_comparison.png")

# =========================================================
# CHART 2: Per-Question Faithfulness
# =========================================================
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(questions_short))
width = 0.32

ax.bar(x - width/2, [v * 100 for v in ollama_faith], width, label="Ollama (llama3.2)", color=colors_ollama)
ax.bar(x + width/2, [v * 100 for v in groq_faith], width, label="Groq (llama-3.3-70b)", color=colors_groq)

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Faithfulness per Question", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(questions_short, fontsize=9)
ax.set_ylim(0, 115)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("chart_faithfulness_per_question.png", dpi=150)
plt.close()
print("Saved chart_faithfulness_per_question.png")

# =========================================================
# CHART 3: Per-Question Context Recall
# =========================================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(x - width/2, [v * 100 for v in ollama_recall], width, label="Ollama (llama3.2)", color=colors_ollama)
ax.bar(x + width/2, [v * 100 for v in groq_recall], width, label="Groq (llama-3.3-70b)", color=colors_groq)

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Context Recall per Question", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(questions_short, fontsize=9)
ax.set_ylim(0, 115)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("chart_context_recall_per_question.png", dpi=150)
plt.close()
print("Saved chart_context_recall_per_question.png")

# =========================================================
# CHART 4: Per-Question BLEU
# =========================================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(x - width/2, [v * 100 for v in ollama_bleu], width, label="Ollama (llama3.2)", color=colors_ollama)
ax.bar(x + width/2, [v * 100 for v in groq_bleu], width, label="Groq (llama-3.3-70b)", color=colors_groq)

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("BLEU Score per Question", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(questions_short, fontsize=9)
ax.set_ylim(0, 115)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("chart_bleu_per_question.png", dpi=150)
plt.close()
print("Saved chart_bleu_per_question.png")

print("\nAll charts generated.")
