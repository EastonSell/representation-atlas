"""Illustrate geometry versus fixed-model intervention; not benchmark evidence.

Python standard library only. No files, random sampling, model loading or data
access. The projection is chosen by hand, not fitted PCA.
"""
import json
import math


def translate(point, offset):
    """Move a two-dimensional point by a shared offset."""
    return (point[0] + offset[0], point[1] + offset[1])


def distance(a, b):
    """Straight-line (Euclidean) distance between two two-dimensional points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cosine(a, b):
    """Cosine is origin-dependent and undefined for a zero vector."""
    denominator = math.hypot(*a) * math.hypot(*b)
    if denominator == 0:
        raise ValueError('Cosine is undefined for a zero vector')
    return (a[0] * b[0] + a[1] * b[1]) / denominator


def readout(point, weights, bias):
    """A fixed downstream linear score, before any thresholding."""
    return weights[0] * point[0] + weights[1] * point[1] + bias


def project_to_x(point):
    """Discard y, potentially merging points with different labels."""
    return (point[0], 0.0)


def main():
    original = [(1.0, 0.0), (0.0, 1.0)]
    shifted = [translate(p, (2.0, 2.0)) for p in original]
    projection_input = [(1.0, 1.0), (1.0, -1.0)]
    projected = [project_to_x(p) for p in projection_input]
    print('ILLUSTRATION only: not benchmark evidence or pretrained-model inference.')
    print(json.dumps({
        'translation': {
            'original': original, 'shifted': shifted,
            'distance_before': distance(*original),
            'distance_after': distance(*shifted),
            'cosine_before': cosine(*original),
            'cosine_after': cosine(*shifted)},
        'fixed_model': {
            'original_scores': [readout(p, (1, 0), -0.5) for p in original],
            'shifted_fixed_readout_scores': [readout(p, (1, 0), -0.5) for p in shifted],
            'shifted_compensated_readout_scores': [readout(p, (1, 0), -2.5) for p in shifted]},
        'projection': {
            'original': projection_input, 'projected': projected,
            'distance_before': distance(*projection_input),
            'distance_after': distance(*projected)}}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
