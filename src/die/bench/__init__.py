"""Cross-agent benchmarking: one environment family, every policy, one number.

The package exists because scoring an agent was, until now, something each test
file did for itself -- ``tests/behavior`` contains four separate private return
helpers with four sets of assumptions. A comparison across *all* agents needs a
single scoring path, or the comparison measures the harness.

Layout:

- :mod:`die.bench.canonical` -- the shared config. Training and evaluation both
  import it so they cannot drift apart.
- :mod:`die.bench.family` -- the seeded environment dataset.
- :mod:`die.bench.policies` -- one uniform policy interface over scripted and
  learned agents.
- :mod:`die.bench.runner` -- the scoring loop.
"""
