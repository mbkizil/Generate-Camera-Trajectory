"""Path to the bundled example recipe (`recipes/example_recipe.json`), used
by each app's "Load example recipe" button so there's always something to
click immediately -- no need to hand-build or track down a recipe file
first before seeing the batch/structure apps do anything interesting.
"""

from __future__ import annotations

from pathlib import Path

EXAMPLE_RECIPE_PATH = Path(__file__).resolve().parent.parent.parent / "recipes" / "example_recipe.json"
