# data/external/

Place any **additional third-party datasets** here that are not fetched
directly from ChEMBL by this pipeline — for example:

- Alternative bioactivity sources (PubChem BioAssay, BindingDB exports)
- Curated literature datasets used for cross-validation against ChEMBL
- Custom in-house screening data formatted to match the expected schema
  (`canonical_smiles`, `standard_value`, `standard_units`, `standard_relation`)

This folder is intentionally empty by default and is excluded from git
tracking (see `.gitignore`) since external datasets are often large or
subject to separate licensing terms. Document the provenance and license of
anything you place here in a short `SOURCE.md` alongside the file.
