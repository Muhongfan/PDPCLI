
## Create a production ready setup with uv 
* `pip install uv`
* `uv init --package` since we already had a project, `uv init project_name --package` to initialize a new project.
* Add the logic file under `src/pdpcli`
* `uv pip install -e .` to excute make the CLI command pdpcli is available with `pdpcli -h`
