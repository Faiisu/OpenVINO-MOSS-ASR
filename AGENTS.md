# Agent Rules

1. Use modular programming with focused modules and clear responsibilities.
2. Before creating a function or module, inspect existing code and dependencies for reusable functionality. Reuse or extend suitable implementations instead of duplicating them.
3. Inspect the folder structure before adding a module. Place it in the appropriate existing package and avoid redundant directories or overlapping responsibilities.
4. Write all code, identifiers, comments, docstrings, and Markdown documentation in English. Communicate with the user in Thai unless instructed otherwise.
5. Take responsibility for assigned work and complete it, including appropriate validation and documentation. Clearly report any remaining blockers and never claim unperformed validation
6. Use only the project-local `.venv` environment. Manage environments and dependencies with `uv`; never install project dependencies into global or system Python.
7. After every change, run the affected workflow in the project-local `.venv` using `uv`. Use real inputs and the actual model for inference changes when available. Report what ran, whether it passed, and any blockers; mocked tests and syntax checks alone do not establish end-to-end success.
8. Every script and notebook that loads a model must unload it after each use, including failures and interruptions. Use `try/finally`, remove all model, pipeline, and GPU tensor references, run garbage collection, and release the CUDA allocator cache on the device used. Reuse the shared cleanup helper. Verify GPU tensor allocation after real inference; CUDA context memory may remain until the process or kernel exits.
