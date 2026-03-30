import os
import re
import pandas as pd

INPUT_CSV = "hse_conditions_qa.csv"
OUTPUT_DIR = "docs"


SECTION_ORDER = {
    "overview": 1,
    "symptoms": 2,
    "tests": 3,
    "treatment": 4,
    "causes": 5,
    "risk_factors": 6,
    "reducing_risks": 7,
    "prevention": 8,
    "living_with": 9,
    "driving": 10,
}


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_filename(name: str) -> str:
    name = clean_text(name).lower()
    name = name.replace("’", "").replace("'", "")
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("_")


def format_answer(answer: str) -> str:
    answer = clean_text(answer)

    lines = answer.split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            continue

        # preserve bullet lines
        if line.startswith("- "):
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)

    # remove trailing blank line
    while formatted_lines and formatted_lines[-1] == "":
        formatted_lines.pop()

    return "\n".join(formatted_lines).strip()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        print("Input QA CSV is empty.")
        return

    required_cols = ["condition_name", "question", "answer", "section", "source_url"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSV: {missing_cols}")

    df["condition_name"] = df["condition_name"].apply(clean_text)
    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(format_answer)
    df["section"] = df["section"].apply(clean_text)
    df["source_url"] = df["source_url"].apply(clean_text)

    df = df[df["condition_name"] != ""].copy()
    df = df[df["question"] != ""].copy()
    df = df[df["answer"] != ""].copy()

    df["section_order"] = df["section"].map(SECTION_ORDER).fillna(999).astype(int)

    grouped = df.groupby("condition_name", sort=True)

    created_files = 0

    for condition_name, group in grouped:
        group = group.copy()
        group = group.sort_values(
            by=["section_order", "section", "question"]
        ).reset_index(drop=True)

        filename = clean_filename(condition_name) + ".md"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {condition_name}\n\n")

            for i, row in group.iterrows():
                f.write("## Question\n")
                f.write(f"{row['question']}\n\n")

                f.write("### Answer\n")
                f.write(f"{row['answer']}\n\n")

                f.write(f"**Section:** {row['section']}\n\n")
                f.write(f"**Source:** {row['source_url']}\n")

                if i != len(group) - 1:
                    f.write("\n\n---\n\n")

        created_files += 1
        print(f"Created: {filepath}")

    print(f"\nDone. Generated {created_files} markdown files in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()