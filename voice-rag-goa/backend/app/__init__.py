"""Voice RAG (Goa Hacker House) backend package.

Import order matters slightly: `config` performs the UTF-8 stdio fix and loads
`.env`, so it is imported first by everything else.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
