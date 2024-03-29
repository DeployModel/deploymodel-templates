FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime
WORKDIR /app

# Requirements
COPY requirements.txt ./requirements.txt
RUN python3 -m pip install -r ./requirements.txt
COPY . /app


# Setting the PYTHONPATH is mandatory
ENV PYTHONPATH=/app 
CMD ["python", "deploy.py"]
