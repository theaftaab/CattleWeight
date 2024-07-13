FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    sudo \
    curl \
    vim \
    build-essential \
    software-properties-common \
    ssh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /cattle_weight

COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend_predict.py"]