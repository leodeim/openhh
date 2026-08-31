# Spec: temperature converter CLI

Build a small Python 3 project (stdlib only):

1. `temps.py` — module with two functions:
   - `c_to_f(celsius: float) -> float`
   - `f_to_c(fahrenheit: float) -> float`
2. `cli.py` — command line tool:
   - `python3 cli.py c2f 100` prints `212.0`
   - `python3 cli.py f2c 32` prints `0.0`
   - unknown command or non-numeric value: print an error to stderr and exit with code 2
3. `test_temps.py` — unittest tests covering both functions including a
   round-trip case, runnable with `python3 -m unittest`
4. `README.md` — how to run the CLI and the tests
