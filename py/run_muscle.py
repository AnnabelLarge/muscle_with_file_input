#!/usr/bin/env python3
"""Align a benchmark dataset with MUSCLE, using the given pair posteriors.

Stages 0 and 1 of the FSA workflow (``prepare.prepare_dataset``) check the
inputs and build the posterior cache, i.e. the folder of deduplicated
posteriors. This script then writes the dataset's ``pair_map.tsv`` once,
and for each family:

    load_family -> write_family_fasta -> muscle -align -postdir <cache>
    -postmap pair_map.tsv -postfamily <family> -> read_msa -> save_msa

and finally ``check_all_saved``. Output, under ``--out``:

    posterior_cache/            stage 0 (reused if it matches the inputs)
    pair_map.tsv                the map for every family run
    muscle/<family>/seqs.fasta  MUSCLE's input, .afa output and log
    msas/<tag>/<family>.fasta   the MSA, rows in the order of in/<family>
    muscle_run.json             options, MUSCLE version, per-family times

Example:
    python py/run_muscle.py --dataset sabre-twi \\
        --posterior-root /mnt/sdb/annabel/2d_posteriors_bali3_ox_prefab_sabre/neuralTKF_sabre \\
        --out out_sabre_twi_muscle --n-workers 8 --jobs 8
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from muscle_posteriors import (FSA_SCORING, read_msa, run_muscle,  # noqa: E402
                               write_family_fasta, write_pair_map)

from common import run_families, write_json  # noqa: E402
from dataset_pairs import DATASETS, family_cost  # noqa: E402
from prepare import (DEFAULT_DATA_DIR, DEFAULT_ONE_ORIENTATION_DIR,  # noqa: E402
                     DEFAULT_REFERENCES_DIR, check_all_saved, load_family,
                     prepare_dataset, save_msa)

DEFAULT_MUSCLE = os.path.join(os.path.dirname(HERE), 'bin', 'muscle')


def muscle_version(muscle_bin):
    p = subprocess.run([muscle_bin, '-version'], capture_output=True,
                       text=True)
    return (p.stdout + p.stderr).strip().splitlines()[0]


def align_family(prep, family, pair_map, muscle_bin, consiters, refineiters,
                 threads, tag):
    """One family: MUSCLE with the given posteriors; returns its record."""
    t0 = time.time()
    fam = load_family(prep, family)
    work = os.path.join(prep['work_dir'], 'muscle', family)
    os.makedirs(work, exist_ok=True)
    fasta = os.path.join(work, 'seqs.fasta')
    afa = os.path.join(work, 'muscle.afa')
    write_family_fasta(fam, fasta)
    run_muscle(fasta, afa, muscle_bin, prep['cache_dir'], pair_map, family,
               consiters=consiters, refineiters=refineiters, threads=threads,
               log=os.path.join(work, 'muscle.log'))
    msa = read_msa(afa, fam)
    path = save_msa(prep, family, msa, tag)
    return {'n_seqs': fam['n_seqs'], 'n_cols': len(next(iter(msa.values()))),
            'msa': os.path.relpath(path, prep['work_dir']),
            'seconds': round(time.time() - t0, 3)}


def run_dataset(dataset, posterior_root, out, muscle_bin=DEFAULT_MUSCLE,
                families=None, consiters=2, refineiters=100, threads=1,
                jobs=1, n_workers=1, tag='muscle',
                data_dir=DEFAULT_DATA_DIR,
                one_orientation_dir=DEFAULT_ONE_ORIENTATION_DIR,
                references_dir=DEFAULT_REFERENCES_DIR):
    """Stages 0-1, the map, MUSCLE on each family. Returns the summary."""
    t0 = time.time()
    if not os.path.exists(muscle_bin):
        raise FileNotFoundError(f'no MUSCLE binary at {muscle_bin}')
    prep = prepare_dataset(dataset, posterior_root, out, data_dir=data_dir,
                           one_orientation_dir=one_orientation_dir,
                           references_dir=references_dir,
                           n_workers=n_workers, families=families)
    pair_map = os.path.join(out, 'pair_map.tsv')
    n_lines = write_pair_map(prep, pair_map)
    jobs_list = [(f, (prep, f, pair_map, muscle_bin, consiters, refineiters,
                      threads, tag), family_cost(prep['records'][f]))
                 for f in prep['families']]
    per_family = run_families(align_family, jobs_list, jobs)
    check_all_saved(prep, tag)
    summary = {
        'dataset': dataset, 'posterior_root': os.path.abspath(posterior_root),
        'posterior_dir': os.path.abspath(prep['cache_dir']),
        'pair_map': os.path.abspath(pair_map), 'pair_map_lines': n_lines,
        'muscle_bin': os.path.abspath(muscle_bin),
        'muscle_version': muscle_version(muscle_bin),
        'fsa_scoring': os.path.abspath(FSA_SCORING),
        'options': {'consiters': consiters, 'refineiters': refineiters,
                    'threads': threads, 'input_order': True},
        'tag': tag, 'families_requested': families,
        'n_families': len(per_family), 'per_family': per_family,
        'seconds': round(time.time() - t0, 3),
    }
    write_json(os.path.join(out, 'muscle_run.json'), summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    p.add_argument('--posterior-root', required=True,
                   help='Folder with SORTED-<SET>-DEDUP/ for each set of the '
                        'dataset, as for run_fsa.py. Only read.')
    p.add_argument('--out', required=True, help='Output directory.')
    p.add_argument('--muscle-bin', default=DEFAULT_MUSCLE)
    p.add_argument('--families', default=None,
                   help='Comma-separated subset of the families. The checks '
                        'and stage 0 still cover the whole dataset.')
    p.add_argument('--consiters', type=int, default=2,
                   help="MUSCLE's -consiters (default 2, MUSCLE's default)")
    p.add_argument('--refineiters', type=int, default=100,
                   help="MUSCLE's -refineiters (default 100, MUSCLE's "
                        'default)')
    p.add_argument('--threads', type=int, default=1,
                   help='MUSCLE threads per family')
    p.add_argument('--jobs', type=int, default=1,
                   help='families aligned at once')
    p.add_argument('--n-workers', type=int, default=1,
                   help='workers for stage 0 (the posterior cache)')
    p.add_argument('--tag', default='muscle', help='msas/<tag>/')
    p.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    p.add_argument('--one-orientation-dir',
                   default=DEFAULT_ONE_ORIENTATION_DIR)
    p.add_argument('--references-dir', default=DEFAULT_REFERENCES_DIR)
    args = p.parse_args(argv)
    families = None
    if args.families:
        families = [s.strip() for s in args.families.split(',') if s.strip()]
    s = run_dataset(args.dataset, args.posterior_root, args.out,
                    muscle_bin=args.muscle_bin, families=families,
                    consiters=args.consiters, refineiters=args.refineiters,
                    threads=args.threads, jobs=args.jobs,
                    n_workers=args.n_workers, tag=args.tag,
                    data_dir=args.data_dir,
                    one_orientation_dir=args.one_orientation_dir,
                    references_dir=args.references_dir)
    print(f'{s["n_families"]} families aligned in {s["seconds"]} s; '
          f'MSAs in {os.path.join(args.out, "msas", args.tag)}')


if __name__ == '__main__':
    main()
