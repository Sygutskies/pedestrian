FROM python:3.11-slim
WORKDIR /pedestrian
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 graphviz xauth
RUN pip install torchview
RUN pip install graphviz
RUN pip install torchinfo
RUN pip install lapx>=0.5.2
CMD ["bash"]