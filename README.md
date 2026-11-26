# Fake Review Detector

Streamlit prototype for classifying reviews with review text, star ratings, and
linguistic features. Predictions and the genuine-likeness score are estimates,
not proof that a review is fake or genuine.

## Data limitations

The included `data/raw/fake_review_Dataset.csv` contains 40 synthetic examples
for exercising the pipeline. It is not representative training data; its
cross-validation scores are only a pipeline check and must not be presented as
real-world performance.

For meaningful evaluation, replace it with a properly licensed, representative,
human-labeled dataset. The CSV must contain:

- `text_`: review text
- `label`: `CG` for fake/generated or `OR` for genuine
- `rating`: numeric value from 1 to 5

The loader rejects unknown labels and invalid ratings rather than silently
mapping them to a class. Account-history and verified-purchase fields are not
used: the current dataset has no labeled examples for those features.

## Setup and run

From the project root, using the project's Windows virtual environment:

```powershell
.\src\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Install dependencies if needed:

```powershell
.\src\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

## Train models

Generate DistilBERT embeddings and linguistic features, then train the Fusion
model:

```powershell
.\src\.venv\Scripts\python.exe .\src\embed.py
.\src\.venv\Scripts\python.exe .\src\fusion.py
```

Fusion and the TF-IDF baseline report stratified out-of-fold metrics, then
refit their saved model on all supplied data. Train the baseline separately:

```powershell
.\src\.venv\Scripts\python.exe .\src\baseline.py
```

Train DML from the processed arrays:

```powershell
.\src\.venv\Scripts\python.exe .\src\dml.py
```

DML reports stratified out-of-fold metrics and calibrates its novelty threshold
from out-of-fold novelty scores. All reported metrics remain unreliable for
real-world claims when using the included synthetic dataset.

## Tests

```powershell
.\src\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The downloadable JSON result summary omits the submitted review text. Cite the
source and license of any real dataset used for experiments.

## Reference

Salminen et al. (2022), “Creating and detecting fake reviews of online products.”
