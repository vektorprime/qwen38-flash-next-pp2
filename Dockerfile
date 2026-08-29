FROM vllm/vllm-openai:nightly
WORKDIR /opt/vllm
ARG PR=53899
RUN apt-get update -qq && apt-get install -y -qq git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "setuptools-scm>=8" "setuptools-rust>=1.9" cmake ninja 2>&1 | tail -5
RUN git init -q && git remote add origin https://github.com/vllm-project/vllm.git && \
    git fetch --depth 1 origin pull/${PR}/head:qwen38 && git checkout qwen38 && \
    git log --oneline -1 && ls vllm/models/qwen4_exp/nvidia/model_state.py
COPY patch_pp.py patch_p0.py /tmp/
RUN python3 /tmp/patch_pp.py && python3 /tmp/patch_p0.py
RUN VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation --no-cache-dir 2>&1 | tail -30 && python3 -c "import vllm; print(vllm.__version__)"
WORKDIR /workspace
ENTRYPOINT ["vllm"]
