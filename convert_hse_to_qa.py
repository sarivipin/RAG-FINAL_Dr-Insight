import re
import pandas as pd

INPUT_CSV = "hse_conditions_long.csv"
OUTPUT_CSV = "hse_conditions_qa.csv"


SECTION_QUESTION_MAP = {
    "overview": "What is {condition_name}?",
    "symptoms": "What are the symptoms of {condition_name}?",
    "tests": "How is {condition_name} diagnosed?",
    "treatment": "What is the treatment for {condition_name}?",
    "reducing_risks": "How can the risk of {condition_name} be reduced?",
    "risk_factors": "Who is at risk of {condition_name}?",
    "driving": "Can I drive if I have {condition_name}?",
    "causes": "What causes {condition_name}?",
    "prevention": "How can {condition_name} be prevented?",
    "living_with": "What is it like to live with {condition_name}?",
}


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


DROP_SECTIONS = {
    "contents",
    "content",
    "support_links",
    "hse_live_were_here_to_help",
    "related_topics",
    "more_information",
}


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_multiline_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]

    cleaned_lines = []
    previous_blank = False

    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()

        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue

        cleaned_lines.append(line)
        previous_blank = False

    return "\n".join(cleaned_lines).strip()


def slugify(text: str) -> str:
    text = clean_text(text).lower()
    text = text.replace("’", "").replace("'", "")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text.strip("_")


def normalize_section(section: str) -> str:
    section = clean_text(section).lower()
    section = section.replace("’", "").replace("'", "")
    section = re.sub(r"[^\w\s-]", "", section)
    section = re.sub(r"\s+", "_", section)

    custom_map = {
        "whos_at_risk": "risk_factors",
        "whos_at_risk_of": "risk_factors",
        "reducing_risk": "reducing_risks",
        "symptom": "symptoms",
        "test": "tests",
    }

    return custom_map.get(section, section)


def is_valid_answer(text: str) -> bool:
    text = clean_multiline_text(text)
    if not text:
        return False

    flat = clean_text(text).lower()

    if len(flat) < 30:
        return False

    junk_values = {
        "contents",
        "overview",
        "treatment",
        "back to top",
        "overview treatment",
        "contents overview treatment",
    }

    if flat in junk_values:
        return False

    return True


def generate_question(condition_name: str, section: str) -> str:
    condition_name_lower = clean_text(condition_name).lower()
    template = SECTION_QUESTION_MAP.get(section)

    if template:
        return template.format(condition_name=condition_name_lower)

    readable_section = section.replace("_", " ")
    return f"What should I know about {condition_name_lower} and {readable_section}?"


def build_chunk_id(condition_name: str, section: str) -> str:
    return f"{slugify(condition_name)}::{section}"


def main():
    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        print("Input CSV is empty.")
        return

    required_cols = ["condition_name", "section", "content", "source_url", "source_path"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSV: {missing_cols}")

    df["condition_name"] = df["condition_name"].apply(clean_text)
    df["section"] = df["section"].apply(normalize_section)
    df["content"] = df["content"].apply(clean_multiline_text)
    df["source_url"] = df["source_url"].apply(clean_text)
    df["source_path"] = df["source_path"].apply(clean_text)

    df = df[df["condition_name"] != ""].copy()
    df = df[df["section"] != ""].copy()
    df = df[~df["section"].isin(DROP_SECTIONS)].copy()
    df = df[df["content"].apply(is_valid_answer)].copy()

    df["question"] = df.apply(
        lambda row: generate_question(row["condition_name"], row["section"]),
        axis=1,
    )
    df["answer"] = df["content"]
    df["chunk_id"] = df.apply(
        lambda row: build_chunk_id(row["condition_name"], row["section"]),
        axis=1,
    )
    df["section_order"] = df["section"].map(SECTION_ORDER).fillna(999).astype(int)

    final_df = df[
        [
            "chunk_id",
            "condition_name",
            "question",
            "answer",
            "section",
            "source_url",
            "source_path",
            "section_order",
        ]
    ].copy()

    final_df = final_df.drop_duplicates(
        subset=["condition_name", "section", "answer"]
    ).copy()

    final_df = final_df.sort_values(
        by=["condition_name", "section_order", "section", "question"]
    ).reset_index(drop=True)

    final_df = final_df[
        [
            "chunk_id",
            "condition_name",
            "question",
            "answer",
            "section",
            "source_url",
            "source_path",
        ]
    ]

    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Total rows: {len(final_df)}")
    print("\nSample output:\n")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()