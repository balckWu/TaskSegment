PYTHON ?= python
CONFIG ?= configs/kbs_3seeds.json

.PHONY: validate plan launch status dashboard watch collect export stats reset compile test prepare-transunet

validate:
	$(PYTHON) scripts/paper.py validate --config $(CONFIG)

plan:
	$(PYTHON) scripts/paper.py plan --config $(CONFIG) --show-commands

launch:
	$(PYTHON) scripts/paper.py launch --config $(CONFIG)

status:
	$(PYTHON) scripts/paper.py status --config $(CONFIG)

dashboard:
	$(PYTHON) scripts/paper.py dashboard --config $(CONFIG)

watch:
	$(PYTHON) scripts/paper.py watch --config $(CONFIG) --interval 20

collect:
	$(PYTHON) scripts/paper.py collect --config $(CONFIG)

export:
	$(PYTHON) scripts/paper.py export --config $(CONFIG)

stats:
	$(PYTHON) scripts/paper.py stats --config $(CONFIG) --reference tasksegment_full

reset:
	$(PYTHON) scripts/paper.py reset --config $(CONFIG)

compile:
	$(PYTHON) -m compileall -q paperlab scripts tasksegment

test:
	$(PYTHON) -m pytest -q tests

prepare-transunet:
	bash scripts/prepare_transunet_pretrained.sh
