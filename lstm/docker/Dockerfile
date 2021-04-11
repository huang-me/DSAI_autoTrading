FROM tensorflow/tensorflow:latest-gpu


COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt
RUN pip install jupyterlab

WORKDIR /home

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]