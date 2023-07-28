.PHONY: start
start:
	python app.py
	
test:
	PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TRANSFORMERS_OFFLINE=1 python test.py

chat:
	python test.py chat

ingest:
	python ingest.py

mlock:
	@echo 'To set new value for mlock, please run: sudo prlimit --memlock=35413752832:35413752832 --pid $$$$'
	prlimit --memlock

.PHONY: format
format:
	isort .
	black .

install:
	pip install -U -r requirements_gpu.txt
	pip show langchain transformers
	
install-extra:
	CXX=g++-11  CC=gcc-11 pip install -U -r requirements_extra.txt
	pip show langchain llama-cpp-python transformers
	
install-extra-mac:
	# brew install llvm libomp
	CXX=/usr/local/opt/llvm/bin/clang++ CC=/usr/local/opt/llvm/bin/clang pip install -U -r requirements_extra.txt
	pip show langchain llama-cpp-python transformers
