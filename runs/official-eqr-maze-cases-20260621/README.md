# Official EqR Maze Case Evidence

This run adds path-aware case visualization for the official EqR e256 clean-vs-FutureSeed pair.

Result: token accuracy is misleading at this budget. Both clean EqR and EqR+FutureSeed have final token accuracy near 0.868 and exact 0, but on the first 64 official test cases both almost never emit PATH. Final loop16 path F1 is about 0.001 for both.

Decision: do not claim Maze is solved or FutureSeed beats EqR from e256. A matched official e512 pair is running on GPU1 to test whether longer official training opens PATH prediction.
