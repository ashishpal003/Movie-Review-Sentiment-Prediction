FROM python:3.12-slim

WORKDIR /app

COPY flask_app/ .

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

RUN ls -la

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 8080

# CMD ["python", "app.py"]
CMD ["gunicorn'"", "--bind", "0.0.0.0:8080", "--timeout", "120", "app:app"]