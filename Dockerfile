FROM python:3.8

COPY . /code
WORKDIR /code

RUN pip3 install -e .

CMD python script.py
