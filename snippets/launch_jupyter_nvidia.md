

::: {.cell .markdown}

## Launch a Jupyter container

Inside the SSH session, build a container image for a Jupyter server with ONNX and related libraries for CPU inference installed:

```bash
# run on node-serve-model 
docker build -t jupyter-onnx -f serve-model-chi/docker/Dockerfile.jupyter-onnx-gpu .
```

Then, launch the container:

```bash
# run on node-serve-model 
docker run  -d --rm  -p 8888:8888 \
    --gpus all \
    --shm-size 16G \
    -v ~/serve-model-chi/workspace:/home/jovyan/work/ \
    -v food11:/mnt/ \
    -e FOOD11_DATA_DIR=/mnt/Food-11 \
    --name jupyter \
    jupyter-onnx
```

To access the Jupyter service, we will need its randomly generated secret token (which secures it from unauthorized access). We'll get this token by running `jupyter server list` inside the `jupyter` container:

```bash
# run on node-serve-model 
docker exec jupyter jupyter server list
```

Look for a line like

```
http://localhost:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of `localhost`, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running *on your compute instance*.

Then, in the file browser on the left side, open the "work" directory and then click on the `4_measure_torch.ipynb` notebook to continue.

:::
