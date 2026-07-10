# 🤝 Contributing to ESC-50 AST Project

Welcome to the project! We appreciate your help in improving this codebase. Before contributing, please review the guidelines below.

---

## 1. Development Workflow

1. **Fork the repository** and create a feature branch from the `main` branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Set up your local environment** as described in [DEVELOPMENT.md](file:///Users/legend27648/agy_project/AI%20Audio/ESC_Project/DEVELOPMENT.md).
3. Make your changes and write automated tests if you introduce new functions or modules.

---

## 2. Coding Standards

* **PEP 8 Compliance**: Enforce PEP 8 conventions. Use a linter or formatter (like `black` or `flake8`) before committing.
* **Strict Type Hints**: All functions must declare argument and return types:
  ```python
  def process_file(self, file_path: str, use_hf: bool = False) -> torch.Tensor:
  ```
* **Documentation**: All public classes, functions, and modules must include complete docstrings outlining parameters and return values.
* **Logging**: Do not use `print()` for debugging in production code. Use the standard logger `logging.getLogger("ESC_Pipeline")`.

---

## 3. Testing Requirements

We enforce a strict **no-regression policy**. Any pull request that modifies code or adds new features must pass all unittests.

* Run the existing tests before submitting your code:
  ```bash
  python -m unittest discover -s tests
  ```
* If you introduce a new feature, you must write corresponding tests inside the `tests/` folder. Follow the formatting used in `test_model.py`.

---

## 4. Pull Request (PR) Checklist

Before submitting a PR, verify that:
1. All unit and integration tests pass successfully.
2. Code conforms to PEP 8 standards and contains complete type hints.
3. No code comments like `TODO` or `FIXME` are left in the source files.
4. You have updated the documentation (`README.md` or other markdown guides) to reflect your changes.
