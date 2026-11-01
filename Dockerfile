FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=5000
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended lmodern \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 resume \
    && mkdir -p /app/data && chown resume:resume /app/data
COPY --chown=resume:resume app ./app
COPY --chown=resume:resume templates ./templates
COPY --chown=resume:resume static ./static
COPY --chown=resume:resume samples ./samples
COPY --chown=resume:resume run.py ./run.py
USER resume
EXPOSE 5000
CMD ["python", "run.py"]
