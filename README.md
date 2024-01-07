# Latent Calendar

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://github.com/wd60622/latent-calendar/actions/workflows/tests.yml/badge.svg)](https://github.com/wd60622/latent-calendar/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/latent-calendar.svg)](https://badge.fury.io/py/latent-calendar)
[![docs](https://github.com/wd60622/latent-calendar/actions/workflows/docs.yml/badge.svg)](https://wd60622.github.io/latent-calendar/)

Analyze and model data on a weekly calendar

## Installation

Install from PyPI: 

```bash
pip install latent-calendar
```

Or install directly from GitHub for the latest functionality. 

## Features 

- Integrated automatically into `pandas` with [`cal` attribute on DataFrames and Series](./modules/extensions.md)
- Compatible with [`scikit-learn` pipelines and transformers](./examples/model/sklearn-compat.md)
- [Transform and visualize data on a weekly calendar](./examples/cal-attribute.md)
- [Model weekly calendar data with a mixture of calendars](methodology.md)
- Create lower dimensional representations of calendar data

## Documentation 

Find more examples and documentation [here](https://wd60622.github.io/latent-calendar/).
