# Use a base image with Python installed
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

RUN pip install --upgrade pip

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Import the vader-lexicon for nltk
RUN python -m nltk.downloader vader_lexicon

# Copy the rest of your application files into the container
COPY . .

# Set environment variables (if necessary)
ENV PYTHONUNBUFFERED=1

# Expose the port for the web server
EXPOSE 5000

# Set the entrypoint to the python command and the orchestrator script
ENTRYPOINT ["python", "app/live/live_orchestrator.py"]
#ENTRYPOINT ["python", "app/live/live_scans_fetcher.py"]

# Set default arguments (can be overridden at runtime)
CMD ["no_terminal"]#, "remote"]
#CMD ["remote"]
#CMD ["sleep", "infinity"]