# Template
Template repository for ASU CS Research organization. Specifies the folder hierarchy expected for downstream utilization
by our CI/CD pipelines. 

## Description:
This is a template repository for the ASU CS Research organization. It is meant to be used as a starting point for new
repositories. It enforces a standardized environment for our auto-documentation system and CI/CD pipelines. 

## Usage:
Follow the steps below to use this template:
1. Modify the `DockerfileDocs` file to include any relevant OS-packages required to install the Python modules for your 
project.
2. Modify the `requirements.txt` file to include any Python modules required for your project. Keep this file up-to-date
during development.
3. Modify the `docs/requirements.txt` file to include any documentation-specific (i.e. Sphinx) Python packages that are 
not required to run your code, but are required for the documentation system.
4. Place your source code in the `src` directory.
5. Place your tests in the `tests` directory.
6. If your project has an API, edit the `openapi.yaml` file to include the API specification, and uncomment
   `.github/workflows/docs.yml` lines `15` and `16` to enable the API documentation workflow.
7. Remember to update the `README.md` file with information relevant to your repository, and remove the default template
`README.md` text.
