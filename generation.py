import json
from datetime import datetime

from retrieval import ask


# =========================================================
# CONFIG
# =========================================================
SAVE_HISTORY = True
HISTORY_FILE = "generation_history.json"


# =========================================================
# HELPERS
# =========================================================
def format_sources(sources):
    if not sources:
        return "No strong sources found."

    lines = []
    for i, src in enumerate(sources, start=1):
        question = src.get("question", "Unknown question")
        disease = src.get("disease", "Unknown disease")
        section = src.get("section", "Unknown section")
        source_url = src.get("source_url", "No source URL")
        score = src.get("rerank_score", 0.0)

        lines.append(
            f"{i}. {question}\n"
            f"   Disease: {disease}\n"
            f"   Section: {section}\n"
            f"   Rerank Score: {score:.4f}\n"
            f"   Source: {source_url}"
        )

    return "\n\n".join(lines)


def save_history(entry, history_file=HISTORY_FILE):
    if not SAVE_HISTORY:
        return

    try:
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
            if not isinstance(history, list):
                history = []
    except FileNotFoundError:
        history = []
    except json.JSONDecodeError:
        history = []

    history.append(entry)

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def build_history_entry(result):
    return {
        "timestamp": datetime.now().isoformat(),
        "query": result.get("query", ""),
        "detected_disease": result.get("detected_disease"),
        "detected_section": result.get("detected_section"),
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }


def display_result(result):
    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(result.get("answer", "No answer generated."))

    print("\n" + "=" * 80)
    print("RETRIEVAL INFO")
    print("=" * 80)
    print(f"Detected disease: {result.get('detected_disease')}")
    print(f"Detected section: {result.get('detected_section')}")

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)
    print(format_sources(result.get("sources", [])))
    print()


# =========================================================
# MAIN
# =========================================================
def main():
    print("=== Medical RAG Generation ===")
    print("Ask a question. Type 'exit' to quit.\n")

    while True:
        query = input("Question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        if not query:
            print("Please enter a question.\n")
            continue

        try:
            result = ask(query)
            display_result(result)

            history_entry = build_history_entry(result)
            save_history(history_entry)

        except Exception as e:
            print("\nError during generation:")
            print(str(e))
            print()


if __name__ == "__main__":
    main()