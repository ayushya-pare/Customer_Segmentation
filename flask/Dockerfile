FROM python:3.12-alpine

RUN apk update && apk add build-base

EXPOSE 5001/tcp
WORKDIR /app

RUN pip install pandas seaborn pandas flask matplotlib

COPY telecom_users.csv .
COPY app.py .

CMD [ "python", "./app.py" ]
