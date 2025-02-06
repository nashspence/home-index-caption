FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    attr \
    file \
    git \
    tzdata \
    rustc \
    && apt-get clean

# run imagemagick easy installer
RUN t=$(mktemp) && \
    wget 'https://dist.1-2.dev/imei.sh' -qO "$t" && \
    bash "$t" && \
    rm "$t"

RUN identify -version

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY packages/home_index_caption .

ENTRYPOINT ["python3", "/app/main.py"]
