# LoanLens: Explainable Loan Approval with Counterfactuals and Policy-Grounded Answers

LoanLens predicts whether a loan application is likely to be approved and explains every decision. It answers three questions an applicant or loan officer actually asks:

1. **Will this loan be approved?** A gradient-boosted classifier (LightGBM, compared with CatBoost) predicts the outcome with a probability.
2. **Why?** SHAP shows which factors pushed the decision up or down for this specific applicant.
3. **What would change it?** DiCE generates counterfactuals: the smallest realistic changes (for example, a lower loan amount or a co-applicant income) that would flip a rejection into an approval.

A retrieval-augmented assistant then answers questions about lending rules, grounded in policy documents and citing the passages it used.

Live demo: http://13.127.240.138:8501/

## Features

- **Prediction.** Approval probability from LightGBM and CatBoost, compared on the same validation split.
- **Local explanations.** SHAP values for each prediction, plus global feature importance.
- **Counterfactuals.** DiCE suggestions limited to features an applicant can realistically change.
- **Policy assistant.** A RAG pipeline (LangChain, Chroma, nomic-embed-text embeddings, a local LLM through Ollama) that answers with sources and streams its answer.
- **Evaluation of the assistant.** RAGAS metrics (faithfulness, answer relevance, context precision) on a fixed question set.
- **Synthetic-data experiment.** Models trained on real data, synthetic data and a mix, compared on the same real test set.
- **Deployment.** A Streamlit app hosted on AWS EC2.

## Results

| Model | Training data | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| LightGBM | Real | <value> | <value> | <value> |
| CatBoost | Real | <value> | <value> | <value> |
| LightGBM | Real + synthetic | <value> | <value> | <value> |

| RAG metric (RAGAS) | Score |
|---|---|
| Faithfulness | <value> |
| Answer relevance | <value> |
| Context precision | <value> |

All scores are on a held-out test set of real applications that no model saw during training.

## How it works

```
Applicant form (Streamlit)
        |
        v
Preprocessing -> LightGBM / CatBoost -> approval probability
        |                 |
        |                 +--> SHAP: why this decision
        |                 +--> DiCE: what would change it
        v
Policy question -> retriever (Chroma, nomic-embed-text, top 3 chunks)
                -> LLM (Ollama) -> answer with cited sources, streamed
```

### Making the assistant fast

The first version took too long to answer on a CPU server. Retrieval was fast; generation was the bottleneck. Five changes brought the first words on screen down to 1-2 seconds:

1. **Shorter context.** 3 retrieved chunks of 600 characters instead of 5 of 1,000.
2. **Bounded answers.** A cap of 150 answer tokens.
3. **Model kept loaded.** `keep_alive` set to 30 minutes, with a warm-up call when the app starts.
4. **Caching.** Repeated questions are answered from a cache.
5. **Streaming.** Answers are streamed with `st.write_stream`, so the first words appear immediately.

## Dataset

[Loan Prediction dataset](https://www.kaggle.com/) (Kaggle).

Features:

- gender, marital status, dependants, education, self-employment;
- applicant and co-applicant income;
- loan amount and term;
- credit history;
- property area.

Target: loan status (approved or not).

## Tech stack

| Area | Tools |
|---|---|
| Modelling | Python, pandas, scikit-learn, LightGBM, CatBoost |
| Explainability | SHAP, DiCE |
| Retrieval and generation | LangChain, Chroma, nomic-embed-text, Ollama |
| Evaluation | scikit-learn metrics, RAGAS |
| App and deployment | Streamlit, AWS EC2 (Amazon Linux) |

## Project structure

```
<folder tree from the repository>
```

## Run locally

```bash
git clone https://github.com/Aishwaryasrivastav31/Loan-lens.git
cd Loan-lens
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull <llm model>
streamlit run <app file>.py
```

The app opens at http://localhost:8501.

## Limitations

- **Counterfactuals are not causal.** They show what would change the model's decision, not what would change a real lender's decision.
- **Explanations, not advice.** The policy assistant answers only from the documents it has indexed and is not financial advice.

## Author

Aishwarya Srivastava · [LinkedIn](https://www.linkedin.com/in/aishwarya-srivastav-5a63222b1/) · [GitHub](https://github.com/Aishwaryasrivastav31)
