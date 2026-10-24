"""Compare retrieval filters: absolute threshold only vs. a margin around the best chunk.

Run from the project folder:  py eval_margin.py
For each question it lists which chunks survive each margin, then scores them:
  recall    - did the chunk that holds the answer survive?
  clean     - did NO unrelated chunk survive?
Add your own cases from eval_retrieval.py to CASES (same idea: question, must-have, must-not-have).
"""

import rag_assistant as ra

# (question, source that must be kept, sources that must NOT be kept; page None = whole file)
CASES = [
    ("How many vacation days do I get?", ("handbook.txt", None), [("company_policy.pdf", 3)]),
    ("How much is the home office stipend?", ("company_policy.pdf", 1), []),
    ("What is the daily meal allowance during travel?", ("company_policy.pdf", 3), []),
    ("Can I carry over unused vacation days?", ("handbook.txt", None), [("company_policy.pdf", 3)]),
]
MARGINS = [None, 0.5, 0.4, 0.35, 0.3, 0.2]


def key(c):
    return (c["source"], c["page"] if c["source"].endswith(".pdf") else None)


def main():
    ra.index_documents()
    print(f"{'margin':>8} | recall | clean | avg chunks kept")
    rows = {m: [0, 0, 0] for m in MARGINS}
    for question, must, must_not in CASES:
        found = ra.retrieve(question, n_results=5)
        print(f"\n{question}")
        for c in found:
            print(f"   {c['distance']:.3f}  {key(c)}")
        for m in MARGINS:
            kept = {key(c) for c in ra.select_relevant(found, margin=m)}
            rows[m][0] += must in kept
            rows[m][1] += not any(x in kept for x in must_not)
            rows[m][2] += len(kept)
    n = len(CASES)
    print()
    for m in MARGINS:
        r, cl, k = rows[m]
        print(f"{str(m):>8} | {r}/{n}    | {cl}/{n}   | {k / n:.1f}")


if __name__ == "__main__":
    main()
