# LoanLens

Explainable loan approval with counterfactual guidance and policy-grounded answers.

**Live application:** [http://13.127.240.138:8501](http://13.127.240.138:8501/)

## Overview

Most credit models return a single outcome, approved or rejected, with no reason and no way forward. That serves neither side of the decision. An applicant needs to know why they were declined and what they could change. A lender needs to justify every decision it makes.

LoanLens is built around that gap. For each application it answers four questions:

| Question | Method |
|---|---|
| Will this loan be approved? | Gradient-boosted classifiers (LightGBM, benchmarked against CatBoost) return an approval probability. |
| Why was this decision made? | SHAP attributes the decision to individual factors such as income, credit history, loan amount and term, for that specific applicant. |
| What would change the outcome? | DiCE generates counterfactual explanations: the smallest realistic changes that would turn a rejection into an approval. |
| What do the lending rules say? | A retrieval-augmented assistant answers policy questions from indexed documents and cites the passages it used. |

## Key capabilities

**Applicant-level explanations.** SHAP explains each individual decision, not only overall feature importance. A loan officer can see exactly which factors raised or lowered this applicant's probability, and by how much.

**Actionable counterfactuals.** DiCE searches only over features an applicant can realistically change, such as the loan amount, the loan term or a co-applicant's income. Fixed attributes are never suggested as changes, so every recommendation is one the applicant could act on.

**Policy assistant with sources.** Policy documents are split into chunks, embedded with nomic-embed-text and stored in Chroma. For each question, the most relevant passages are retrieved through LangChain and passed to a local language model served by Ollama. The answer is grounded in those passages and shows its sources.

**Measured answer quality.** The assistant is evaluated with RAGAS on faithfulness, answer relevance and context precision, so its quality is measured rather than assumed.

**Synthetic-data study.** Models trained on real data, synthetic data and a mix of the two are compared on the same held-out set of real applications. This tests whether synthetic data improves a small credit dataset.

**Deployed application.** The full system runs as a Streamlit application on AWS EC2.

## Engineering: a fast assistant on a CPU server

The first deployed version answered policy questions slowly. Profiling showed that retrieval took milliseconds and the delay came from text generation on CPU. Long prompts made it worse, and so did the model being unloaded after five idle minutes. Five targeted changes brought the time to the first visible words down to one to two seconds.

| Change | Effect |
|---|---|
| Context reduced from five 1,000-character chunks to three 600-character chunks | Shorter prompt, faster generation |
| Answer length capped at 150 tokens | Predictable response time |
| Model kept in memory for 30 minutes, with a warm-up call at start-up | No cold-start delay |
| Cache for repeated questions | Instant answers to common queries |
| Streamed output | Text appears as it is generated |

## Dataset

Loan Prediction dataset (Kaggle).

| Item | Detail |
|---|---|
| Applicant profile | Gender, marital status, dependants, education, self-employment |
| Financial features | Applicant income, co-applicant income |
| Loan details | Loan amount, loan term, property area |
| Credit history | Whether the applicant has a credit history |
| Target | Loan status (approved or not) |

## Technology

| Area | Tools |
|---|---|
| Data and modelling | Python, pandas, NumPy, scikit-learn, LightGBM, CatBoost |
| Explainability | SHAP, DiCE |
| Retrieval-augmented generation | LangChain, Chroma, nomic-embed-text, Ollama |
| Evaluation | scikit-learn metrics, RAGAS |
| Application and deployment | Streamlit, AWS EC2 |

## Running locally

```bash
git clone https://github.com/Aishwaryasrivastav31/Loan-lens.git
cd Loan-lens
pip install -r requirements.txt

ollama pull nomic-embed-text
ollama pull phi3

streamlit run app.py
```

The application opens at http://localhost:8501.

## Author

Aishwarya Srivastava
[LinkedIn](https://www.linkedin.com/in/aishwarya-srivastav-5a63222b1/) | [GitHub](https://github.com/Aishwaryasrivastav31)
