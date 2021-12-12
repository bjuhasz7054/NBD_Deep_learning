FROM tensorflow/tensorflow:2.7.0-gpu

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pipenv

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY ./src ./src
COPY ./bin ./bin
COPY ./Pipfile .
COPY ./Pipfile.lock .
COPY config.ini .

RUN pipenv install --system --deploy --ignore-pipfile --clear
RUN rm -rf /root/.cache/pipenv

ENV PYTHONPATH=/usr/src/app

CMD ["bash"]
