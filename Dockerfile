# Create a Dockerfile for the FastAPI application

# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# The command to run your application will be in docker-compose.yml
# But it's good practice to specify a default command here
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]