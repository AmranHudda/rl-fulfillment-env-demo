FROM python:3.12-slim
WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt && useradd --create-home controller
COPY environment.py controller_server.py mcp_server.py container_entrypoint.py ./
USER controller
EXPOSE 7000 8000
ENTRYPOINT ["python3", "container_entrypoint.py"]
