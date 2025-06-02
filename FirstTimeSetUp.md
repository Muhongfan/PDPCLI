
## Create a production ready setup with uv 
* `pip install uv`
* `uv init --package` since we already had a project, `uv init project_name --package` to initialize a new project.
* Add the logic file under `src/pdpcli`
* `uv venv .venv` Add a new venv named as `.venv`
* `source .venv/bin/activate` to active the venv
* `uv pip install -e .` to install `pdpcli` under `.venv/bin/`.
* Check the avaliable CLI command pdpcli `pdpcli -h`

