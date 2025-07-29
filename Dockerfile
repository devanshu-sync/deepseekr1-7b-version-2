FROM python:3.12-slim

WORKDIR /

# Install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your handler file
COPY main.py /

# Start the container
CMD ["python3", "-u", "main.py"]