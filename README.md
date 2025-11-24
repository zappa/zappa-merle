# merle README

This project uses zappa/ollama to deploy ollama compatable models to AWS lambda.

## CLI Usage

The CLI provides commands for preparing, deploying, and managing Ollama model deployments:

```bash
# Prepare deployment files for an Ollama model
python -m merle.cli prepare --model {OLLAMA_MODEL} [--auth-token TOKEN] [--tags KEY=VALUE,...]

# Deploy a prepared model to AWS Lambda
python -m merle.cli deploy --model {MODEL_NAME} --auth-token {AUTH_TOKEN}

# List all configured models
python -m merle.cli list

# Start an interactive chat session with a deployed model
python -m merle.cli chat --model {MODEL_NAME}

# Tear down a deployed Lambda function
python -m merle.cli destroy --model {MODEL_NAME}
```

**Note:** You can find a list of available Ollama models at [https://ollama.com/library](https://ollama.com/library)

### AWS Configuration

Before deploying, ensure your AWS credentials are configured. Merle uses the standard AWS credential chain:

```bash
# Option 1: Set AWS profile (recommended for multiple accounts)
export AWS_PROFILE=your-profile-name

# Option 2: Set credentials directly
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key

# Optional: Set default region (overrides the CLI default)
export AWS_DEFAULT_REGION=us-east-1
```

**Region Configuration:**
- Default region: `ap-northeast-1`
- Override with `--region` option: `merle prepare --model llama2 --region us-west-2`
- Or set via environment: `export AWS_DEFAULT_REGION=us-west-2`

**Note:** The region must be specified during `prepare` step as it's embedded in the deployment configuration.

### Using uvx (Recommended)

You can run `merle` without installing it using [uvx](https://docs.astral.sh/uv/guides/tools/), which executes the CLI in an isolated environment:

```bash
# Prepare deployment files (with optional region)
uvx merle prepare --model llama2 --auth-token YOUR_TOKEN --region us-east-1

# Deploy to AWS Lambda
uvx merle deploy --model llama2 --auth-token YOUR_TOKEN

# List configured models
uvx merle list

# Start interactive chat
uvx merle chat --model llama2

# Destroy deployment
uvx merle destroy --model llama2

# Check version
uvx merle --version
```

**Benefits of using uvx:**
- No installation required
- Always uses an isolated environment
- Fast subsequent runs due to caching
- Perfect for CI/CD pipelines and one-off commands

**Note:** First run may take a moment to set up the environment, but subsequent runs are nearly instant due to uv's caching.

## Structure

```
zappa-merle/
├── .github/
│   └── workflows/
│       ├── register-circleci-project.yml
│       └── test.yml
├── merle/
│   ├── __init__.py
│   ├── app.py
│   ├── chat.py
│   ├── cli.py
│   ├── functions.py
│   ├── settings.py
│   └── templates/
│       ├── Dockerfile.template
│       ├── authorizer.py
│       └── zappa_settings.json.template
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_chat.py
│   ├── test_cli.py
│   ├── test_deployment_completeness.py
│   ├── test_docker.py
│   └── test_functions.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
```

## Local Development

Python: 3.13

> Requires [uv](https://docs.astral.sh/uv/guides/install-python/) for dependency management


### Installing Development Environment

1. Install `pre-commit` hooks (_ruff_):

    > Assumes [pre-commit](https://pre-commit.com/#install) is already installed.

    ```bash
    pre-commit install
    ```

2. The following command installs project and development dependencies:

    ```bash
    uv sync
    ```

## Run Code Checks

```
uv run poe check
```

Run type checking:
```
uv run poe typecheck
```

## Run Test Cases

This project uses [pytest](https://docs.pytest.org/en/latest/contents.html) for running testcases.

Test cases should be added in the `tests` directory.

To run test cases, execute the following command:
```
pytest -v
# Or, from the parent directory
uv run poe test
```

