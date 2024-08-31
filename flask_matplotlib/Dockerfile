FROM python:3.12-slim


EXPOSE 5001/tcp
WORKDIR /app

RUN pip install pandas seaborn pandas flask matplotlib scikit-learn
COPY telecom_users.csv .
COPY app.py .
COPY model.pkl .
COPY feature_names.pkl .

CMD [ "python", "./app.py" ]
