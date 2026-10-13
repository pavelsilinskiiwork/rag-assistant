from rag_assistant import THRESHOLD, index_documents, retrieve

index_documents()

# (question, should the answer be found in the documents?)
TEST_CASES = [
    ("How many vacation days do I get?", True),
    ("Can I carry over unused vacation days?", True),
    ("Can I work from home on Fridays?", True),
    ("Do I need approval to buy a $150 keyboard?", True),
    ("What should I do if I lose my laptop?", True),
    ("What is the CEO's salary?", False),
    ("What is the company's parental leave policy?", False),
    ("Who won the football game yesterday?", False),
]

passed = 0
for question, should_find in TEST_CASES:
    best = retrieve(question, n_results=1)[0]
    found = best["distance"] <= THRESHOLD
    ok = found == should_find
    passed += ok
    status = "OK  " if ok else "FAIL"
    print(f"{status} | distance={best['distance']:.3f} | expected_found={should_find} | {question}")

print(f"\nPassed {passed}/{len(TEST_CASES)}")