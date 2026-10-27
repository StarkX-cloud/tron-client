"""The routing layer: decides where a job should actually run.

The first thing in this repo that connects two previously separate
systems: ``tron_efficiency`` (single-node memory virtualization — can this
machine run it, and how) and ``tron``'s distributed spine/matcher (can the
pooled fleet of other machines run it). Neither package is modified to
depend on the other; this package imports from both and sits above them,
so each remains independently usable the way it was before.

See ``decision.py``'s module docstring for the real, honest scope of what
"routing" means here — this picks a target (local / pooled / dedicated),
it does not provision or guarantee anything about the target it picks.
"""
