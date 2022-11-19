from .hake import build as build_hake


def build_dataset(split, args):
    return build_hake(split, args)
