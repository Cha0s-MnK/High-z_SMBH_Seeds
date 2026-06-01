# McLuster Workflow for NSC + Two-BH Initial Conditions

This README describes a simple workflow for generating nuclear star cluster (NSC) initial conditions with [McLuster](https://github.com/lwang-astro/mcluster), then adding two massive black holes (BHs): one central primary BH and one less massive secondary BH initially placed near the core on a circular orbit.

The intended science case is to test whether the secondary BH sinks to the centre, stalls in a cored NSC, or forms a bound binary with the central BH.

## 1. Install McLuster

Clone the repository:

```bash
git clone https://github.com/lwang-astro/mcluster.git
cd mcluster
```

Build the basic version:

```bash
make clean
make mcluster
```

Check that the executable works:

```bash
./mcluster -h
```

For this workflow, the basic `mcluster` executable is enough. You do not need the SSE/BSE version unless you want McLuster to evolve the stellar population before outputting the initial condition.

## 2. McLuster Output Used in This Workflow

Use McLuster only to generate the stellar NSC background. The two BH particles are added afterward by post-processing.

For this workflow, use table-output mode:

```bash
-C 3 -u 1
```

where:

```text
-C 3   only create an output list of stars
-u 1   output in astrophysical units
```

The output file has the form:

```text
<output_name>.txt
```

with columns:

```text
Mass_[Msun]  x_[pc]  y_[pc]  z_[pc]  vx_[km/s]  vy_[km/s]  vz_[km/s]
```

The last two BH particles should later be appended in the same column format.

## 3. Recommended Case Design and Initial Matrix

Use three families of models:

| Case family | Purpose | McLuster profile |
|---|---|---|
| Cored NSC | Main Banik-style core-stalling/buoyancy test | EFF/Nuker with inner slope 0 |
| Cuspy NSC | Control case where sinking should be easier | Nuker with inner slope 1 |
| Central-BH-dominated NSC | Test whether a large primary BH destroys the clean core-stalling regime | Same stellar model as cored NSC, but larger central BH |

For the first controlled experiment, use single-mass stellar particles instead of a full IMF. This makes the dynamical-friction and particle-noise behavior easier to interpret.

A useful first set of cases is:

| Label | Stellar model | Primary BH mass `M1` | Secondary BH mass `M2` | Initial secondary radius `r0` | Purpose |
|---|---:|---:|---:|---:|---|
| `core_q0.1_r1` | cored | `1e4 Msun` | `1e3 Msun` | `1 pc` | fiducial stalling test |
| `core_q1_r1` | cored | `1e4 Msun` | `1e4 Msun` | `1 pc` | strong secondary |
| `core_q0.1_r2` | cored | `1e4 Msun` | `1e3 Msun` | `2 pc` | inspiral then possible stall |
| `cusp_q0.1_r1` | cuspy | `1e4 Msun` | `1e3 Msun` | `1 pc` | cuspy control |
| `core_BHdom_r1` | cored | `1e5 Msun` | `1e3 Msun` | `1 pc` | central-BH-dominated limit |

A simple starting choice for the stellar NSC is:

```text
M_NSC  ~ 1e6 Msun
N_star = 100000
m_star ~ 10 Msun effective particles
r_core ~ 1 pc
r_cut  = 20 pc
```

## 4. Generate the Stellar NSC with McLuster

### 4.1 Cored NSC

Use an EFF/Nuker-like profile with a flat inner slope:

```bash
./mcluster \
  -N 100000 \
  -f 0 \
  -P 3 \
  -r 1.0 \
  -c 20.0 \
  -g 4.0 -g 0.0 -g 2.0 \
  -Q 0.5 \
  -C 3 \
  -u 1 \
  -s 1001 \
  -o nsc_core
```

Meaning:

```text
-N 100000              number of stellar particles
-f 0                   single-mass stars
-P 3                   EFF/Nuker profile
-r 1.0                 scale/break radius in pc
-c 20.0                cutoff radius in pc
-g 4.0 -g 0.0 -g 2.0   outer slope, inner slope, transition sharpness
-Q 0.5                 virial equilibrium
-C 3                   output only the star table
-u 1                   astrophysical units
-s 1001                fixed random seed
-o nsc_core            output prefix
```

This produces:

```text
nsc_core.txt
```

The important part for the cored model is:

```bash
-g 4.0 -g 0.0 -g 2.0
```

where the second value is the inner slope. Setting it to `0.0` gives a flat central core.

### 4.2 Cuspy Control Model

Generate the cuspy comparison model by changing only the inner slope from `0.0` to `1.0`:

```bash
./mcluster \
  -N 100000 \
  -f 0 \
  -P 3 \
  -r 1.0 \
  -c 20.0 \
  -g 4.0 -g 1.0 -g 2.0 \
  -Q 0.5 \
  -C 3 \
  -u 1 \
  -s 1001 \
  -o nsc_cusp
```

This produces:

```text
nsc_cusp.txt
```

The only intended structural difference from the cored model is:

```bash
-g 4.0 -g 1.0 -g 2.0
```

where the inner slope is now `1.0`.

## 5. Add the Two BHs and Generate the Actual IC Cases

After generating the stellar NSC table, add the two BH particles manually through a small post-processing script.

The appended file should keep the same seven-column format:

```text
Mass_[Msun]  x_[pc]  y_[pc]  z_[pc]  vx_[km/s]  vy_[km/s]  vz_[km/s]
```

Use the convention:

```text
BH1: central primary BH at the NSC centre
BH2: secondary BH at radius r0 with circular velocity
```

The circular velocity of the secondary should be computed as:

```text
v_circ(r0) = sqrt( G * [M1 + M_star(<r0)] / r0 )
```

with:

```text
G = 0.00430091733 pc km^2 s^-2 Msun^-1
```

Recommended post-processing steps:

1. Read the McLuster stellar table.
2. Recenter the stellar system to its centre of mass.
3. Compute the stellar enclosed mass `M_star(<r0)`.
4. Place BH1 at the centre.
5. Place BH2 at `(x, y, z) = (r0, 0, 0)`.
6. Assign BH2 velocity `(vx, vy, vz) = (0, v_circ, 0)`.
7. Append BH1 and BH2 as the last two rows.
8. Recenter the full system after adding the BHs.
9. Save the final IC table.

Assume the helper command has the form:

```bash
./add_two_bhs.py <input_stellar_table> <output_ic_table> --m1 <M1> --m2 <M2> --r0 <r0>
```

### 5.1 Fiducial Cored Case

```bash
./add_two_bhs.py nsc_core.txt ic_core_M1e4_M2e3_r1.txt \
  --m1 1.0e4 \
  --m2 1.0e3 \
  --r0 1.0
```

### 5.2 Equal-Mass BH Case

```bash
./add_two_bhs.py nsc_core.txt ic_core_M1e4_M2e4_r1.txt \
  --m1 1.0e4 \
  --m2 1.0e4 \
  --r0 1.0
```

### 5.3 Larger Initial Radius Case

```bash
./add_two_bhs.py nsc_core.txt ic_core_M1e4_M2e3_r2.txt \
  --m1 1.0e4 \
  --m2 1.0e3 \
  --r0 2.0
```

### 5.4 Cuspy Control Case

```bash
./add_two_bhs.py nsc_cusp.txt ic_cusp_M1e4_M2e3_r1.txt \
  --m1 1.0e4 \
  --m2 1.0e3 \
  --r0 1.0
```

### 5.5 Central-BH-Dominated Case

```bash
./add_two_bhs.py nsc_core.txt ic_core_M1e5_M2e3_r1.txt \
  --m1 1.0e5 \
  --m2 1.0e3 \
  --r0 1.0
```

## 6. Suggested Directory Layout

A compact layout is:

```text
mcluster/
├── mcluster
├── add_two_bhs.py
├── nsc_core.txt
├── nsc_cusp.txt
├── ic_core_M1e4_M2e3_r1.txt
├── ic_core_M1e4_M2e4_r1.txt
├── ic_core_M1e4_M2e3_r2.txt
├── ic_cusp_M1e4_M2e3_r1.txt
└── ic_core_M1e5_M2e3_r1.txt
```

The files beginning with `nsc_` are pure stellar NSC models generated by McLuster. The files beginning with `ic_` are the final NSC + two-BH initial conditions.
