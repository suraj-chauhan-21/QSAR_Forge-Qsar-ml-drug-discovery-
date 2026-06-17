# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Yes              |
| < 1.0   | ❌ No               |

## Reporting a Vulnerability

This is a research/educational codebase for QSAR machine learning modeling. It
does not handle secrets, user authentication, or production infrastructure,
but we still take security seriously — particularly around:

- Arbitrary code execution via unpickled model files (`joblib`/`pickle` loading)
- Dependency vulnerabilities (RDKit, XGBoost, LightGBM, CatBoost, PyTorch, etc.)
- Unsafe deserialization of untrusted ChEMBL/SMILES input

### How to Report

If you discover a security vulnerability, please **do not open a public GitHub
issue**. Instead:

1. Open a [private security advisory](../../security/advisories/new) on GitHub
   (preferred), or
2. Email the maintainer with details (see repository profile for contact)

Please include:
- A description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept if possible)
- Any suggested remediation

### What to Expect

- **Acknowledgment**: within 5 business days
- **Initial assessment**: within 10 business days
- **Fix or mitigation timeline**: communicated once the issue is triaged

We'll credit reporters in the release notes unless anonymity is requested.

## Known Considerations

- **Model files** (`.pkl`, `.pt`, `.cbm`) loaded via `joblib.load()` or
  `torch.load()` can execute arbitrary code if the file is untrusted. Only
  load model artifacts from sources you trust.
- **ChEMBL API calls** fetch data over HTTPS from `www.ebi.ac.uk`; no
  credentials are required or stored.
- Dependencies are pinned with minimum versions in `requirements.txt` /
  `environment.yml`; run `pip list --outdated` periodically and consult
  `pip-audit` or GitHub's Dependabot alerts for known CVEs.
