# syntax=docker/dockerfile:1.2
FROM python:3.9-slim

# Install make and other necessary tools
RUN apt-get update && apt-get install -y make

# Set the working directory to the root of the project
WORKDIR /latam-challenge

# Copy the requirements files and Makefile
COPY requirements.txt requirements-dev.txt requirements-test.txt Makefile ./

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Copy the model file
COPY model.pkl .

# Expose the port FastAPI will run on
EXPOSE 8080

# Set the entry point to run the FastAPI application
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]