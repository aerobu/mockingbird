# Synthetic stanza parody generation using llama 3.1 via Ollama
import json
import subprocess
from tqdm import tqdm
import re
import nltk
import syllables
from nltk.util import ngrams

nltk.download("punkt")


stanzaFile = "stanzas_test.txt"
outputJsonl = "parodies_llama31_scored.jsonl"
themes = ["mathematics", "soccer", "geography"]
variantsPerTheme = 2
ollamaModel = "llama3.1"
maxTokens = 200


def line_squared_error(groundtruth_line, generated_line):
    gt = syllables.estimate(groundtruth_line)
    gen = syllables.estimate(generated_line)
    return (gen - gt) ** 2


def calculate_mse_of_syllable_counts(groundtruth_text, generated_text, delimiter='\n'):
    gt_lines = groundtruth_text.split(delimiter)
    gen_lines = generated_text.split(delimiter)

    # pad to equal length
    max_len = max(len(gt_lines), len(gen_lines))
    gt_lines += [''] * (max_len - len(gt_lines))
    gen_lines += [''] * (max_len - len(gen_lines))

    errors = [line_squared_error(gt, gen) for gt, gen in zip(gt_lines, gen_lines)]
    mse = sum(errors) / len(errors)
    return mse


def calculate_unique_ngram_percentage(text, n=2):
    tokens = nltk.word_tokenize(text.lower())
    generated_ngrams = list(ngrams(tokens, n))
    if len(generated_ngrams) == 0:
        return 0.0
    unique_ngrams = set(generated_ngrams)
    return len(unique_ngrams) / len(generated_ngrams)


def truncate_stanza(stanza, lines_in_original_stanza, delimiter='\n'):
    lines = stanza.split(delimiter)
    return delimiter.join(lines[:lines_in_original_stanza])


def filter_empty_lines(stanza, delimiter='\n'):
    lines = stanza.split(delimiter)
    filtered = [l for l in lines if l.strip() != ""]
    return delimiter.join(filtered)


def call_ollama(prompt):
    cmd = ["ollama", "run", ollamaModel]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = process.communicate(prompt)
    return out.strip()


def load_stanzas(path):
    stanzas = []
    current = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            if line.lower().startswith("stanza"):
                if current:
                    stanzas.append("\n".join(current))
                    current = []
            else:
                if line.strip():
                    current.append(line)

        if current:
            stanzas.append("\n".join(current))

    return stanzas


stanzas = load_stanzas(stanzaFile)


def clean_stanza(output, expected_lines):
    lines = [l.strip() for l in output.split("\n") if l.strip()]

    # If too many lines → take top N
    if len(lines) > expected_lines:
        lines = lines[:expected_lines]

    # If too few → pad
    if len(lines) < expected_lines:
        lines = lines + [""]*(expected_lines - len(lines))

    return "\n".join(lines)


with open(outputJsonl, "w", encoding="utf-8") as out_f:
    for stanza_id, stanza in enumerate(tqdm(stanzas, desc="Processing stanzas"), start=1):
        original_line_count = len([l for l in stanza.split("\n") if l.strip()])

        for theme in THEMES:
            best_parody = None
            best_score = None
            best_ngram_score = None

            for variant in range(1, variantsPerTheme + 1):
                # STRONG prompt to keep model aligned
                prompt = f"""
Write a parody of the stanza below.

STRICT RULES:
- Output EXACTLY {original_line_count} lines.
- The parody MUST clearly reflect the theme: "{theme}".
- NO explanation.
- NO reasoning.
- NO extra commentary.
- ONLY output the rewritten stanza.

Original stanza:
{stanza}

Parody (exactly {original_line_count} lines):
""".strip()

                raw_output = call_ollama(prompt)

                cleaned = clean_stanza(raw_output, original_line_count)

                original_truncated = truncate_stanza(stanza, original_line_count)
                parody_truncated = truncate_stanza(cleaned, original_line_count)

                mse = calculate_mse_of_syllable_counts(original_truncated, parody_truncated)
                ngram_score = calculate_unique_ngram_percentage(parody_truncated, n=2)

                total_score = (-mse, ngram_score)  # lexicographically sorted

                if best_score is None or total_score > best_score:
                    best_score = total_score
                    best_parody = parody_truncated
                    best_ngram_score = ngram_score

            out_f.write(json.dumps({
                "stanza_id": stanza_id,
                "theme": theme,
                "best_parody": best_parody,
                "syllable_mse": -best_score[0],       # positive MSE
                "unique_bigram_pct": best_ngram_score,
                "original": stanza
            }, ensure_ascii=False) + "\n")
