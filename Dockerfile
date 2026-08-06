FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY code ./code
COPY configs ./configs
COPY dataset_links.txt .
ENV PYTHONPATH=/workspace/code
ENTRYPOINT ["python", "-m", "equityfedhcc.commands.train"]

