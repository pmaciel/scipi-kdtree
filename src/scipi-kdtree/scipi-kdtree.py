#!/usr/bin/env python3


import argparse
import pickle
from os import environ, makedirs, path

import eccodes
import numpy as np
from scipy.spatial import KDTree


def ll_to_ecef(lat, lon):
    lonr = np.radians(lon)
    latr = np.radians(lat)

    x = np.cos(latr) * np.cos(lonr)
    y = np.cos(latr) * np.sin(lonr)
    z = np.sin(latr)
    return x, y, z


def point(str):
    try:
        lat, lon = map(int, str.split(","))
        return (lat, lon)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid point format: '{str}'. Expected format: lat,lon"
        )


def main(args=None):
    parser = argparse.ArgumentParser(description="GRIB file nearest point search")

    parser.add_argument("input", help="GRIB file")
    parser.add_argument(
        "--point",
        type=point,
        nargs="+",
        required=True,
        help="List of points in the format lat,lon",
    )

    parser.add_argument("--no-caching", dest="caching", action="store_false")

    search = parser.add_mutually_exclusive_group()
    search.add_argument("--nclosest", help="Search number", type=int)
    search.add_argument("--distance", help="Search radius (on unit sphere)", type=float)

    args = parser.parse_args(args)
    print(args)

    assert args.nclosest or args.distance

    with open(args.input, "rb") as f:
        h = eccodes.codes_grib_new_from_file(f)
        assert h

        # k-d tree
        tree_dir = environ["TMPDIR"] if "TMPDIR" in environ else "."
        tree_path = path.join(
            tree_dir, eccodes.codes_get(h, "md5GridSection") + ".tree"
        )

        if args.caching and path.exists(tree_path):
            print(f"Loading cache file: '{tree_path}'")
            with open(tree_path, "rb") as f:
                tree = pickle.load(f)
        else:
            N = eccodes.codes_get(h, "numberOfDataPoints")
            it = eccodes.codes_grib_iterator_new(h, 0)

            P = np.empty([N, 3])
            i = 0
            while True:
                result = eccodes.codes_grib_iterator_next(it)
                if not result:
                    break
                [lat, lon, value] = result

                assert i < N
                P[i, :] = ll_to_ecef(lat, lon)

                i += 1

            eccodes.codes_grib_iterator_delete(it)
            tree = KDTree(P)

        if args.caching and not path.exists(tree_path):
            makedirs(tree_dir, mode=888, exist_ok=True)
            assert path.isdir(tree_dir)
            with open(tree_path, "wb") as f:
                pickle.dump(tree, f)
            print(f"Created cache file: '{tree_path}'")

        for lat, lon in args.point:
            distances, indices = (
                tree.query(ll_to_ecef(lat, lon), k=args.nclosest)
                if args.nclosest
                else tree.query_ball_point(ll_to_ecef(lat, lon), r=args.distance)
            )

            indices.sort()  # Sort indices to make output relatable to scanningMode
            print(indices)


if __name__ == "__main__":
    main()
