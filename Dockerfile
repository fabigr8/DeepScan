FROM python:3.9

# add folders
RUN mkdir models
ENV MODEL_DIR=/models
ENV MODEL_FILE_IF=clf_if.joblib
ENV MODEL_FILE_NN=clf_nn.joblib
ENV MODEL_FILE_COPOD=clf_copod.joblib
ENV MODEL_FILE_PIPEL=full_pipeline.joblib
ENV PYTHONPATH=/deepscanapp

# install packages
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# copy code
COPY /deepscanapp ./deepscanapp

#add env variables


# setworkdir
WORKDIR /deepscanapp

# start univcorn
EXPOSE 8000
# ENTRYPOINT ["uvicorn"]
# , "--port", "8000"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0"]
