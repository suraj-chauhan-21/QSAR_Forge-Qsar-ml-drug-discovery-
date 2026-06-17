# Contributing to QSAR-ML Drug Discovery

Thanks for your interest in contributing! This project welcomes contributions of all kinds:
bug fixes, new models, additional featurization methods, documentation improvements, and more.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/QSAR-ML-Drug-Discovery.git
   cd QSAR-ML-Drug-Discovery
   ```
3. **Set up the environment**:
   ```bash
   conda env create -f environment.yml
   conda activate qsar-ml
   pip install -e ".[dev]"
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

All new functionality should include corresponding unit tests in `tests/`.

### Code Style

This project uses `black`, `isort`, and `flake8` for code formatting and linting.

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
```

These checks run automatically in CI on every pull request.

### Commit Messages

Use clear, descriptive commit messages in the imperative mood:
- ✅ `Add Mordred descriptor support`
- ✅ `Fix unit conversion bug for picomolar values`
- ❌ `updates`
- ❌ `fixed stuff`

## Submitting a Pull Request

1. Push your branch to your fork
2. Open a Pull Request against the `main` branch of this repository
3. Fill out the PR template, describing what changed and why
4. Ensure all CI checks pass (tests, lint)
5. Respond to review feedback

## Areas We'd Love Help With

- **New models**: Graph Neural Networks (GNNs), Transformer-based molecular models, Gaussian Processes
- **New featurization**: Mordred descriptors, 3D conformer-based descriptors, learned embeddings (e.g., ChemBERTa)
- **Classification support**: Binary active/inactive prediction alongside regression
- **Scaffold-based splitting**: Bemis-Murcko or Butina clustering for more rigorous train/test splits
- **Web interface**: A Streamlit or Gradio app for interactive compound screening
- **Additional ChEMBL targets**: Pre-configured example notebooks for other disease-relevant targets
- **Documentation**: Tutorials, docstring improvements, translated READMEs

## Code of Conduct

Be respectful, constructive, and patient. This is a learning-oriented, community project — questions
and "naive" PRs are welcome. We're all here to get better at computational drug discovery together.

## Questions?

Open an issue, or reach out via the [Omixium YouTube channel](https://www.youtube.com/@omixium) comments.
