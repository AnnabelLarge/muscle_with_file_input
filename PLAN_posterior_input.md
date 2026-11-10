# Plan: MUSCLE with posteriors read from files

Goal: `muscle -align` keeps every step after the pairwise posteriors exactly as
it is now (sparse matrices, EA distances, UPGMA guide tree, consistency,
progressive alignment, refinement). Only the source of each pair's dense
posterior changes: MUSCLE's own pair HMM, or a float32 matrix taken from two
dataset-wide inputs:

1. **the folder of deduplicated posteriors**: one `.npy` per kept pair,
   deduplicated across families and orientations. This is the stage-0 cache
   that the Python pipeline of `../msa_with_new_decoder/fsa_scoring_on_ada_new_decoder/`
   already builds and checks (`<work_dir>/posterior_cache/SORTED-<S>-DEDUP/<kept_pairID>.npy`),
   used as it is, with no copy;
2. **a mapping file** (plain TSV) that gives, for each pair of each family,
   the file to use and its orientation. It is written from the stage-1
   records (`prepare.prepare_dataset`), with no matrix read.

All of the checking, deduplication lookup and the choice of orientation stay
in Python. The C++ side reads the map, looks up a line by (family, labels),
reads one float32 matrix and transposes it if the line says so.

Why a dedup folder + map, not one copy per family: in the real data many pairs
have no posterior of their own; they use the posterior of a pair with the same
two sequences, often in another family and in the other orientation. Counts
from `DATA_msa_benchmarks/SORTED-<S>_dedup_map.tsv` (FW pairs mapped to a row
of another family): BALI3-SHORT 32,132 of 146,305; BALI3-MEDIUM 1,949 of
13,993; OX-t-subset 14,399 of 24,562; OX-s-subset 4,302 of 12,807;
BBS-TRUNC-SHORT 2,646 of 136,958; PREFAB4REF 18 of 1,681; SABRE-SUP, SABRE-TWI,
OX-N 0. MUSCLE's own derep only merges identical sequences within one family,
so it does not remove this need.

---

## 1. Where MUSCLE uses the posteriors (current code)

`cmd_align` (src/align.cpp) → `MPCFlat::Run` (src/mpcflat.cpp):

```
Derep (drop identical sequences)   → unique seqs, in input order
InitPairs                           → pairs (i, j), i < j, in that order
CalcPosteriors  (OpenMP over pairs) → MPCFlat::CalcPosterior(PairIndex)
CalcGuideTree   (UPGMA on m_DistMx)
ClustalWeights  (computed, then overwritten with 1.0)
Consistency     (m_ConsistencyIterCount, default 2)
ProgressiveAlign
Refine          (m_RefineIterCount, default 100; uses libc rand())
SortMSA, InsertDupes
```

`MPCFlat::CalcPosterior` (src/calcposteriorflat.cpp) is the only place the HMM
enters:

```cpp
float *Post = CalcPost(LabelX, LabelY);          // dense LX*LY, row-major, X = rows
SparsePost.FromPost(Post, LX, LY);               // keeps P >= 0.01
Score = CalcAlnScoreFlat(Post, LX, LY, DPRows);  // MEA score on the dense matrix
m_DistMx[X][Y] = m_DistMx[Y][X] = Score / min(LX, LY);
```

Everything downstream reads only the sparse matrices and `m_DistMx`. So if we
swap `CalcPost(...)` for "load the dense float32 matrix for (LabelX, LabelY)",
the rest of MUSCLE runs unchanged. Because the stored matrix holds the same
float32 bits as `Post`, a dump of MUSCLE's own posteriors read back in must
give a byte-identical MSA.

Orientation: X is always the earlier sequence in MUSCLE's input order. Derep
keeps the first copy of each sequence in input order (checked in
src/derep.cpp: `m_RepSeqIndexes` grows in input order), and `InitPairs` uses
i < j. So if the FASTA we pass is in `fam['names']` order, X is the name with
the lower index in `names`.

Facts checked in the code (2026-11-10):

- MUSCLE's label is the whole FASTA header line after `>` (no cut at
  whitespace, unless `-accs`). The reference `parse_fasta` cuts at
  whitespace, so names from `fam['names']` never contain any.
- `LoadMFA(..., stripGaps=true)` removes `-` and `.` and uppercases.
- Derep compares sequences case-insensitively.
- `CreateStdioFile` / `OpenStdioFile` update a global map, so they must not be
  called inside the OpenMP loop; the `.npy` reader/writer uses plain stdio.

## 2. Inputs (per dataset, plus one FASTA per family)

```
<postdir>/                    the stage-0 cache, as built (input 1)
  SORTED-<S>-DEDUP/<kept_pairID>.npy   float32, C order, (anc_len, desc_len)
<pair_map.tsv>                one per dataset (input 2)
<famdir>/seqs.fasta           per family, sequences in fam['names'] order
```

`pair_map.tsv` (tab-separated, one header line):

```
family  row_label  col_label  LX  LY  file  transpose
```

- One line per unordered pair of each family; families in sorted order, pairs
  in the stage-1 order, so two maps of the same inputs are byte-identical.
- `row_label` is the name that comes first in `fam['names']` (MUSCLE's X),
  `col_label` the other; `LX`, `LY` their lengths.
- `file` is relative to `<postdir>`. Several lines can name the same file
  (pairs that share a deduplicated posterior, in one family or in several).
- `transpose` 0: the file has shape `(LX, LY)` (the posterior's ancestor is
  `row_label`). `transpose` 1: shape `(LY, LX)` (ancestor is `col_label`);
  MUSCLE uses its transpose.
- Built from each family's stage-1 provenance: key `(i, j) = (idx[ancestor],
  idx[descendant])`, `file = cache_file`; i < j gives transpose 0, i > j
  gives row = names[j], col = names[i], transpose 1.

The C++ reader accepts only `.npy` format 1.0/2.0 with `descr '<f4'`,
`fortran_order False`, 2-D shape and an exact file size (about 60 lines, no new
dependency). Every cache file of sabre-twi and prefab passes this check (§5).

## 3. C++ changes (small, all in `MPCFlat`)

New options in `src/myopts.h`: `STR_OPT(postdir)`, `STR_OPT(postmap)`,
`STR_OPT(postfamily)` and `STR_OPT(postout)`. `-postdir`, `-postmap` and
`-postfamily` go together (`Die` if only some are set).

1. **`-postout DIR`** (dump the built-in posteriors, in the same format).
   After `CalcPost` in `CalcPosterior`, write `Post` to
   `DIR/pairs/<i>_<j>.npy` (i < j = indexes in the original input, i.e. the
   global sequence index `GetGSIByLabel`), and buffer a map line per
   PairIndex with `transpose` 0 and `family` = `-postfamily` if set, else
   `-`. `CalcPosteriors` writes `DIR/pair_map.tsv` after the OpenMP loop, in
   PairIndex order, so it is deterministic whatever the thread count, and
   `DIR/seqs.fasta` (the parsed input, one line per sequence). A dump is read
   back with `-postdir DIR -postmap DIR/pair_map.tsv -postfamily <same>`.
2. **`-postdir D -postmap M -postfamily F`** (use the given posteriors).
   Before `CalcPosteriors`, read `M` once, keep only the lines of family `F`
   in `map<pair<string,string>, entry>`, and check, serially, for every pair
   (LabelX, LabelY) MUSCLE will need:
   - a line with row = LabelX and col = LabelY exists. If only the swapped
     line exists, `Die` (the orientation is the Python side's job); if none,
     `Die` naming the pair. `Die` if family `F` has no lines, on a repeated
     pair, a bad header, a field count other than 7 or `transpose` not 0/1;
   - LX and LY match the sequence lengths MUSCLE parsed.
   Then in `CalcPosterior`, replace `CalcPost(LabelX, LabelY)` with
   `LoadPost(...)`, which reads `D/file` (shape must be `(LX, LY)`, or
   `(LY, LX)` if `transpose` is 1) into an `AllocPost(LX, LY)` buffer,
   transposing if needed, `Die`s on a non-finite or negative value, and sets
   values `< MIN_SPARSE_PROB` (0.01) to 0, as `CalcPostFlat` does (§7). The
   rest of `CalcPosterior` is unchanged.
3. **Guards when `-postdir` is set:** `Die` if `-perturb` > 0, `-stratified`,
   `-diversified` or `-replicates` is set (these change only the HMM
   parameters, so with fixed posteriors they would silently do nothing).
   `Die` if Mega input is loaded, or if the run would go to Super5
   (`-minsuper`). `-perm`, `-consiters`, `-refineiters`, `-input_order` and
   `-threads` behave as usual. The options apply to `-align` only.
4. Log a line `Posteriors: built-in | from D (map M, family F)` so each run
   records which path it took.

Pairs between duplicate sequences of one family are never needed: MUSCLE
dereps first and looks up only pairs of representatives. Lines of family `F`
that are not used are ignored and counted in the log. Reading the whole map
per family run is cheap (largest dataset about 150k lines).

Build: add `src/Makefile.offline` (same flags and source list as the Makefile
that `build_linux.bash` generates; the sources come from `muscle.vcxproj`,
which leaves out `alpha.cpp` and `help.cpp`, so "all `*.cpp`" would not
link). It cannot be called `src/Makefile`: `vcxproj_make.py` writes that file
and `src/.gitignore` ignores it. New `.cpp` files (`npy.cpp`) are added to
`muscle.vcxproj`.

## 4. Python side (in this repo: `py/`)

`py/muscle_posteriors.py` (written and tested, §5), which imports the
reference workflow unchanged (`$FSA_SCORING_DIR`, default
`../msa_with_new_decoder/fsa_scoring_on_ada_new_decoder`):

- `family_map_rows(family, record)`: the map lines of one family from its
  stage-1 record (§2). Stops if names or provenance disagree, a pair is
  repeated or missing, or a name is empty or contains whitespace.
- `write_pair_map(prep, path, families=None)`: the dataset's map, families
  sorted, written via a temp file and rename.
- `read_pair_map(path)`, `read_npy_strict(path)` and
  `load_pairs_from_map(pair_map, post_dir, family, names)`: read the inputs
  back the way MUSCLE will (same checks, transpose applied), for the tests.
- `family_fasta_seqs(fam)` / `write_family_fasta(fam, path)`: `seqs.fasta`
  built from `fam['int_seqs']` (the `vendored.scoring.encode_seq` encoding
  the posteriors were indexed by), in `fam['names']` order, decoded as
  `AA_CHARS[k]` for k < 20 and `'X'` for the wildcard 20. Lowercase input
  comes out uppercase, every non-standard amino acid becomes `X`, gap
  characters are already removed, so the lengths match the map.

Still to write:

- `run_muscle(famdir, out_afa, muscle_bin, post_dir, pair_map, family,
  consiters=2, refineiters=100, threads=1, extra=())`: subprocess with
  `-align famdir/seqs.fasta -postdir post_dir -postmap pair_map -postfamily
  family -output out_afa -input_order`; raises if MUSCLE exits non-zero.
- `read_msa(out_afa, fam)`: labels equal `fam['names']`, each row without
  gaps equals `seqs.fasta`; returns `{name: row}` in `names` order, and
  `prepare.save_msa` runs the reference `check_msa` against `raw_seqs` (it
  accepts `X` for an input `B`, `Z` or `X`; extend `AMBIGUOUS_TO_WILDCARD`
  there if U, O or J ever appear).
- `py/run_muscle.py`, a CLI: `prepare_dataset → write_pair_map (once) → for
  family: load_family → write_family_fasta → run_muscle → read_msa →
  save_msa(tag='muscle') → check_all_saved`. `load_family` is still called
  per family for its checks and `raw_seqs`; its matrices are not passed on.

## 5. Tests (`py/tests/`, pytest; C++ binary path from `$MUSCLE_BIN`)

Done (`py/tests/test_pair_map.py`, preprocessing only, no MUSCLE run; 7 passed
in 8 s on 2026-11-10). For each family checked, the matrices read back
through the map (`load_pairs_from_map`) equal, bit for bit, the ones
`load_family` gives (transposed for keys i > j), and `seqs.fasta` has the
map's lengths, only the 20 letters plus X, and matches `raw_seqs` residue for
residue under the reference rule.

- toy layout (`tests/layout.py` of the reference workflow, `write_layout`
  unchanged): three bali3 families where BB2 reuses BB1's sequences under
  other names and order (its pairs use BB1's posteriors, in both
  orientations), BB1 has lowercase and B/Z, BB3 two identical sequences.
  Also: the map is byte-identical whatever the family order passed, and
  `transpose` is 1 exactly when the posterior's ancestor is `col_label`.
- real, neuralTKF posteriors on `/mnt/sdb/annabel/2d_posteriors_bali3_ox_prefab_sabre`
  (`$FSA_REAL_POSTERIORS`; skipped when absent), stage 0 + 1 run for the
  whole set:
  - sabre-twi: 108 families, 1,565 map lines, 725 transposed, 1,565
    distinct files, none shared; the 5 smallest families checked;
  - prefab: 1,681 families (pairs), 610 transposed, 1,663 distinct files,
    18 shared between families; the 5 smallest families and 3 sharing
    families checked;
  - every cache file named by either map passes `read_npy_strict`.

To do, with the C++ side:

1. **Baseline determinism (before any change):** the same input run twice
   gives identical `.afa`, with `-threads 1` and `-threads 8`. (Passed on
   29aa067 in the reverted first attempt.) Generated data: random protein
   sequences from `np.random.default_rng(seed)`, families of N = 2, 3, 5 and
   12, lengths 20–300, one with two identical sequences, some related (point
   mutations plus indels from one root).
2. **C++ round trip:** `-postout D` then `-postdir D -postmap D/pair_map.tsv
   -postfamily F`, `cmp` the two `.afa`. Repeated for `-consiters {0,2}`,
   `-refineiters {0,100}`, `-perm {none,abc}` and `-threads {1,8}`. The dump
   is also checked against a second `-postout` run (identical bytes).
3. **Through Python, with dedup and orientation:** from a `-postout` dump of
   several families, build a shared dedup folder in which half of the
   matrices are stored transposed and pairs with equal sequences (within and
   across families) share one file, plus its map; the MSA must be
   byte-identical to the built-in run.
4. **Through the real stage 0–1 pipeline:** the toy layout of §5 (done) with
   MUSCLE's dumps as the posterior samples (`write_layout(posterior=...)`
   accepts given matrices, keyed by kept pairID; no change to that repo), then
   `prepare_dataset → write_pair_map → run_muscle`; the MSA must equal the
   built-in one.
5. **Failure tests:** a missing pair, a family not in the map, the wrong
   shape, a wrong `transpose`, NaN, a negative value, swapped orientation, a
   wrong label, a missing file, or `-postdir` with `-perturb 1`. Each must
   make MUSCLE exit non-zero with a clear message.
6. **Smoke test on real data:** a few small sabre-twi and prefab families
   (including sharing ones) with real posteriors. The MSAs are written and
   pass `check_msa`. Compare scores with the FSA progressive outputs (no
   identity expected).

## 6. Order of work

1. Build MUSCLE (`Makefile.offline`), run test 1, record the git hash and
   timings.
2. `-postout` + the npy writer, then check a dump against `CalcPost`.
3. `-postdir/-postmap/-postfamily` + the npy reader (with transpose) +
   guards, then tests 2 and 5.
4. Preprocessing (`muscle_posteriors.py`, done), then `run_muscle`,
   `read_msa`, tests 3 and 4.
5. `run_muscle.py` on real data (test 6). Add a short README section.

## 7. Decisions (agreed 2026-11-10; inputs changed the same day)

- **Inputs: the dedup folder + a TSV map.** MUSCLE reads the stage-0 cache
  as it is and a dataset-wide map; no per-family copies of the matrices.
  The map says which file and whether to transpose; the transpose is the
  only orientation work in C++, and the choice is made in Python.
- **Randomness: no change to MUSCLE.** Refinement keeps using libc `rand()`
  unseeded, as upstream. With default options, MUSCLE's own generator
  (`randu32`, seeded by `-randseed`, default 1) is only used to guess
  DNA vs. protein, and HMM perturbation is off (perturb seed 0). So a default
  run is deterministic, and the identity tests need no seed handling.
- **Options: keep MUSCLE's default behaviour.** The new input supports the
  default single run. Options that only perturb the HMM (`-perturb`,
  `-stratified`, `-diversified`, `-replicates`) are rejected with it (§3.3).
  No ensemble support is added.
- **Derep: keep MUSCLE's.** Identical sequences are merged before alignment and
  reinserted afterwards, as upstream. Map lines of pairs between duplicates
  are unused. This differs from the FSA pipeline (`test_no_dedup.py`).
- **Values below 0.01: zero them on load.** In `LoadPost`, set `P = 0` where
  `P < MIN_SPARSE_PROB` (0.01). This mirrors `CalcPostFlat`, so built-in and
  loaded matrices are handled the same way in the EA distance for the guide
  tree. (Dumps are already zeroed, so the identity tests are unaffected.)
  Note that `CalcPostFlat` thresholds in log space (`Score <
  logf(0.01)`), so a value within float rounding of 0.01 could be treated
  differently. The tests will show if this ever matters.
- **Python code lives in this repo, under `py/`.** It imports the FSA workflow
  by path.
- **Build: add `src/Makefile.offline`**, and keep the upstream scripts (§3).
- **Input characters:** lowercase letters are converted to uppercase, and every
  non-standard amino acid maps to `X`. Both happen in Python, via the
  `encode_seq` encoding (§4), so the FASTA that MUSCLE reads matches the
  posteriors residue for residue. A scan of every `references/*/in` file
  (3,005 files) found no lowercase letters and no whitespace in names. The
  only non-standard letters were B (365), Z (201) and X (1,381).
  Note: the built-in MUSCLE run in the tests reads the same converted FASTA,
  so both runs see identical input.
- **No conversion of posterior code to C++ is needed.** The C++ side gains
  only an `.npy` reader/writer (with transpose), a map lookup by
  (family, labels) and the 0.01 threshold.

## 8. Status (2026-11-10): all steps done

Branch `posterior-input`. Built with `src/Makefile.offline`; all 156 tests in
`py/tests` pass in 16 s (`MUSCLE_BIN=bin/muscle python -m pytest py/tests`).

| plan test | file | result |
|---|---|---|
| 1 baseline determinism | `test_baseline.py` | 12 passed (1 and 8 threads, with options) |
| 2 C++ round trip | `test_roundtrip_cpp.py` | 77 passed: 4 families × 16 option sets byte-identical; dump does not depend on threads; re-dump of what was read is identical |
| 3 shared dedup folder | `test_dedup_store.py` | 5 passed (see deviations) |
| 4 real stage 0–1 pipeline | `test_pipeline.py` | 12 passed: 5 toy bali3 families byte-identical to built-in |
| 5 failures | `test_failures_cpp.py` | 39 passed |
| 6 real data | `test_real_smoke.py` | 4 passed (scores below) |
| preprocessing | `test_pair_map.py` | 7 passed |

Test 6, mean MSA SP / TC (core columns) over the families checked, MUSCLE
vs. FSA progressive with the same settings (`ci2_ri100`, neuralTKF):
prefab (9 families of 2 sequences) 0.590 / 0.590 for both (identical per
family, as expected for pairwise MEA); sabre-twi (6 families of 3–4
sequences) 0.413 / 0.239 vs. 0.405 / 0.239. Rescoring FSA's saved MSAs
reproduces its `scores.json`.

Deviations from the text above, found while implementing:

- **Orientation and MUSCLE's own HMM.** MUSCLE's posterior of (b, a) is not
  bit for bit the transpose of (a, b): on n12, relative differences up to
  1.5e-3 where both are nonzero, and 1 entry of 22,875 zeroed in one only.
  So test 3 compares a family with its built-in run only when every shared
  pair keeps MUSCLE's orientation. Every family is also compared with a run on
  a plain folder holding the matrices the map implies (no sharing, no
  transpose), which must give the same bytes. Given posteriors are not
  affected: the map defines the orientation.
- **Duplicate order in the output.** With `-input_order`, MUSCLE still puts a
  duplicate right after its first copy (`InsertDupes` runs after `SortMSA`).
  So `read_msa` checks that the labels are `fam['names']`, each once, and
  returns the rows in `names` order. `prepare.save_msa` does not run
  `check_msa`, so `read_msa` calls it.
- **Extra guards with `-postdir`:** `-hmmin` and the transition options
  (`-s_is -s_il -m_is -m_il -is_is -il_il`) are refused too, since they also
  change only the HMM. `-postout` may be combined with `-postdir`, which
  re-dumps what was read after the 0.01 threshold, but it must name another
  directory. `-postout` is refused with Mega input, Super5 and multi-run
  options.
- **NaN check.** The build uses `-ffast-math`, under which GCC may fold
  `std::isfinite` to true, so `LoadPost` tests the exponent bits.
- **Threading.** MUSCLE's `Ps`/`CreateStdioFile` helpers keep global state, so
  the OpenMP loop uses only plain stdio and `std::string`.
- **Threshold.** No dump of the test families had a nonzero value below 0.01,
  so the log-space vs. linear threshold (§7) did not matter in any test.
