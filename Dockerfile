FROM python:3.10-slim

WORKDIR /home/redis_app

COPY ./src /home/redis_app

RUN pip install --no-cache-dir -r /home/redis_app/requirements.txt

EXPOSE 5005

CMD ["uvicorn","redis_helper:app","--host","0.0.0.0","--port","5005"]
