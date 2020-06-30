#!/usr/bin/env python3
import streamlit as st
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')

# The following lines are needed on the first run only
from cdqa.utils.download import download_model  
download_model('bert-squad_1.1')
directory = '/home/antony/u4.3/cdQA/data'
download_model('bert-squad_1.1', dir=directory)
download_model('distilbert-squad_1.1', dir=directory)
from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline.cdqa_sklearn import QAPipeline
from nltk import tokenize

#read the csv text file
def load_from_csv(file):
    df = pd.read_csv(file)
    df = df.rename(str.lower, axis='columns')
    df['paragraphs'] = df['paragraphs'].apply(lambda x: x.replace("'s", " " "s").replace("\n"," "))
    df['paragraphs'] = df['paragraphs'].apply(lambda x: tokenize.sent_tokenize(x))
    return df

df = load_from_csv('./data/test.csv')

#make sure bert_qa.joblib is the same directory (cdQA), if not move it here from data
cdqa_pipeline = QAPipeline(reader='bert_qa.joblib')
cdqa_pipeline.fit_retriever(df=df)

querry=st.text_area('enter mssage','type')
if st.button('analyze'):
	message=cdqa_pipeline.predict(query=querry, n_predictions=2)
	st.success(message)
