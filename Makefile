PY ?= .venv/bin/python
G  ?= .venv/bin/grounded
ABLATION := bm25 bm25_tuned bm25_rm3 bm25_chunk100 dense_lsa hybrid_lsa_rrf hybrid_lsa_tuned

.PHONY: install test lint data tune bench gate verify serve docker

install:
	uv venv .venv && uv pip install -p .venv -e '.[dev]'

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

data:
	$(G) data --dest data/scifact

tune: data
	$(G) eval tune --split train

bench: data  ## offline retrieval ablation on SciFact dev (~30 s on a laptop)
	$(G) eval retrieval $(foreach c,$(ABLATION),--config configs/$(c).yaml) \
	  --out reports/retrieval_scifact_dev.json --report reports/retrieval_scifact_dev.md

bench-neural: data  ## needs `pip install -e '.[neural]'`
	$(G) eval retrieval --config configs/bm25_tuned.yaml --config configs/hybrid_bge.yaml \
	  --config configs/hybrid_bge_rerank.yaml --out reports/retrieval_neural.json \
	  --report reports/retrieval_neural.md

gate:
	$(G) eval gate --results reports/retrieval_scifact_dev.json --thresholds eval/thresholds.yaml

verify: data  ## end-to-end claim verification with Claude; needs ANTHROPIC_API_KEY
	$(G) eval verify --config configs/verify.yaml --limit $(or $(LIMIT),100) --out reports/verify_scifact_dev.json

serve:
	$(G) index --config configs/bm25_tuned.yaml --scifact data/scifact --out indexes/bm25_tuned.pkl
	$(G) serve --index indexes/bm25_tuned.pkl

docker:
	docker build -t grounded-rag .
