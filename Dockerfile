ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.12-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/riiid
WORKDIR /workspace/riiid

RUN pip install --no-cache-dir -r requirements.txt
RUN jupyter serverextension enable --py jupyter_http_over_ws
RUN mkdir -p $(jupyter --data-dir)/nbextensions &&\
    cd $(jupyter --data-dir)/nbextensions &&\
    git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding &&\
    jupyter nbextension enable vim_binding/vim_binding
#RUN jupyter labextension install jupyterlab_vim

ENV PORT=8888
EXPOSE 8888

CMD ["bash", "/workspace/riiid/run_colab.sh"]
