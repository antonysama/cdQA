FROM python:3.6-slim-buster
#ENV PYTHONUNBUFFERED 1
#RUN apt update -y
RUN mkdir /app
     
WORKDIR /app
     
COPY requirements.txt /app/
     
RUN pip3 install --no-cache-dir -r requirements.txt
#RUN python -m nltk.downloader punkt
COPY ./ /app/
     
EXPOSE 8080
     
CMD python app.py
