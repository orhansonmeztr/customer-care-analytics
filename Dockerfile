FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[dashboard]"
COPY dashboard ./dashboard

ENV CARE_DATA_DIR=/data CARE_SCALE=0.3 CARE_FOLDS=6
VOLUME /data
EXPOSE 8501

# Build the data on first start (synthetic, reproducible), then serve the dashboard.
CMD ["sh", "-c", "[ -f $CARE_DATA_DIR/warehouse.duckdb ] || care run-all --scale $CARE_SCALE --folds $CARE_FOLDS; \
     streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
