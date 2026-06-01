# Running McLuster NSC + Two-BH Initial Conditions with PeTar

This guide describes how to run the NSC + two-BH systems generated with McLuster and the BH-insertion script using PeTar.

The target experiment is to test whether a secondary massive BH in a nuclear star cluster sinks to the central BH, stalls in a core, or forms a bound BH binary.

> **Important scope note:** PeTar is a collisional, no-softening N-body code with tree, Hermite, and SDAR components. It is well suited for close encounters and binary formation, but it is not a brute-force direct summation code for all long-range forces. For this project, use PeTar to test orbital decay, stalling, and binary formation. A final gravitational-wave merger requires extra compact-object/PN or stellar-evolution/GW treatment and should not be assumed from pure Newtonian dynamics alone.

---

## 1. Recommended directory layout

Use one parent directory for all related codes:

```bash
mkdir -p ~/codes
cd ~/codes
```

The intended layout is:

```text
~/codes/
├── FDPS/
├── SDAR/
├── PeTar/
└── mcluster/
```

Putting `FDPS`, `SDAR`, and `PeTar` in the same parent directory allows PeTar's configure script to detect the dependencies automatically.

---

## 2. Install PeTar

### 2.1 Clone dependencies

```bash
cd ~/codes

git clone https://github.com/FDPS/FDPS.git
cd FDPS
git checkout v7.0
cd ..

git clone https://github.com/lwang-astro/SDAR.git
```

PeTar currently depends on FDPS and SDAR. FDPS v7.0 is recommended because newer FDPS versions may cause issues in PeTar.

### 2.2 Clone PeTar

```bash
cd ~/codes
git clone https://github.com/lwang-astro/PeTar.git
cd PeTar
```

### 2.3 Configure a CPU-only pure-dynamics version

For this BH-pair experiment, start with a simple pure-dynamics build:

```bash
./configure \
  --prefix=$HOME/tools/petar \
  --with-mpi=auto
```

This keeps the build simple and avoids stellar-evolution treatment of the artificial BH particles.

Check the configuration summary carefully. If `FDPS` and `SDAR` are not automatically detected, specify them manually:

```bash
./configure \
  --prefix=$HOME/tools/petar \
  --with-mpi=auto \
  --with-fdps-prefix=$HOME/codes/FDPS \
  --with-sdar-prefix=$HOME/codes/SDAR
```

### 2.4 Compile and install

```bash
make -j 8
make install
```

Add PeTar to your environment:

```bash
export PATH=$PATH:$HOME/tools/petar/bin
export PYTHONPATH=$PYTHONPATH:$HOME/tools/petar/include
```

For permanent setup, add the two `export` lines to `~/.bashrc`.

### 2.5 Test the installation

```bash
which petar
petar -h
petar.init -h
petar.data.gether -h
petar.data.process -h
```

---

## 3. Input data format

PeTar starts from a PeTar-style snapshot file. The intermediate particle table should contain 7 columns:

```text
m  x  y  z  vx  vy  vz
```

For this project, the McLuster + BH-insertion workflow produces files like:

```text
ic_core_M1e4_M2e3_r1.txt
ic_core_M1e4_M2e4_r1.txt
ic_core_M1e4_M2e3_r2.txt
ic_cusp_M1e4_M2e3_r1.txt
ic_core_M1e5_M2e3_r1.txt
```

These files are assumed to use:

```text
mass      Msun
position  pc
velocity  km/s
```

However, PeTar with `-u 1` expects:

```text
mass      Msun
position  pc
velocity  pc/Myr
```

Therefore, always convert the velocity unit with:

```bash
-v kms2pcmyr
```

when using `petar.init`.

---

## 4. Recommended simulation cases

Use the following first-run matrix:

| Case label | Stellar model | Primary BH `M1` | Secondary BH `M2` | Initial secondary radius `r2` | Purpose |
|---|---:|---:|---:|---:|---|
| `core_q0p1_r1` | cored NSC | `1e4 Msun` | `1e3 Msun` | `1 pc` | fiducial stalling test |
| `core_q1_r1` | cored NSC | `1e4 Msun` | `1e4 Msun` | `1 pc` | strong secondary |
| `core_q0p1_r2` | cored NSC | `1e4 Msun` | `1e3 Msun` | `2 pc` | inspiral then possible stall |
| `cusp_q0p1_r1` | cuspy NSC | `1e4 Msun` | `1e3 Msun` | `1 pc` | cuspy control |
| `core_BHdom_r1` | cored NSC | `1e5 Msun` | `1e3 Msun` | `1 pc` | central-BH-dominated limit |

Expected input files:

```text
ic_core_M1e4_M2e3_r1.txt
ic_core_M1e4_M2e4_r1.txt
ic_core_M1e4_M2e3_r2.txt
ic_cusp_M1e4_M2e3_r1.txt
ic_core_M1e5_M2e3_r1.txt
```

---

## 5. Convert one IC file to PeTar format

Create a run directory:

```bash
mkdir -p runs/core_q0p1_r1
cd runs/core_q0p1_r1
```

Copy the IC file into the run directory:

```bash
cp /path/to/ic_core_M1e4_M2e3_r1.txt ./ic.txt
```

Convert it to PeTar format:

```bash
petar.init -v kms2pcmyr -f input ic.txt
```

This creates a PeTar input snapshot named:

```text
input
```

Do not use stellar-evolution options here. The artificial BHs should remain fixed-mass N-body particles.

---

## 6. Run a first short PeTar test

Use a short integration first to check stability:

```bash
export OMP_STACKSIZE=128M
export OMP_NUM_THREADS=8
ulimit -s unlimited

petar -u 1 \
  -t 1.0 \
  -o 0.05 \
  -w 2 \
  -a 0 \
  input \
  &> output.log
```

Meaning:

```text
-u 1      use Msun, pc, Myr unit system; velocity is pc/Myr
-t 1.0    run to 1 Myr
-o 0.05   output every 0.05 Myr
-w 2      print all particle data in one line with status; useful for small tests
-a 0      overwrite previous output files instead of appending
input     PeTar-format input snapshot from petar.init
```

Inspect the log:

```bash
less output.log
```

Check for:

```text
large energy error
large angular-momentum drift
extreme wall-clock imbalance
crashes in hard/SDAR parts
```

If the short test is stable, run a longer model.

---

## 7. Run a production-scale model

For the BH-pair problem, useful first choices are:

```bash
petar -u 1 \
  -t 20.0 \
  -o 0.1 \
  -a 0 \
  input \
  &> output.log
```

This runs for 20 Myr and writes snapshots every 0.1 Myr.

For a longer stalling test:

```bash
petar -u 1 \
  -t 100.0 \
  -o 0.5 \
  -a 0 \
  input \
  &> output.log
```

The exact runtime should be chosen based on the NSC crossing time, secondary BH sinking time, and computational cost.

---

## 8. Run with MPI + OpenMP

If PeTar was compiled with MPI, run for example:

```bash
export OMP_STACKSIZE=128M
export OMP_NUM_THREADS=8
ulimit -s unlimited

mpiexec -n 4 --bind-to none petar -u 1 \
  -t 20.0 \
  -o 0.1 \
  -a 0 \
  input \
  &> output.log
```

This uses:

```text
4 MPI ranks
8 OpenMP threads per rank
32 total CPU threads
```

Avoid using too many MPI ranks for small-N tests. For dense systems, the hard/SDAR part can become load-imbalanced if one compact subsystem dominates the short-range work.

---

## 9. Example SLURM job script

Create `run_petar.slurm`:

```bash
#!/bin/bash
#SBATCH -J petar_core_q0p1_r1
#SBATCH -p tyhcnormal
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

module purge
module load gcc
module load mpi/openmpi/openmpi-4.1.5-gcc9.3.0

export PATH=$HOME/tools/petar/bin:$PATH
export PYTHONPATH=$HOME/tools/petar/include:$PYTHONPATH

export OMP_STACKSIZE=128M
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -s unlimited

mpiexec -np $SLURM_NTASKS --bind-to none petar -u 1 \
  -t 20.0 \
  -o 0.1 \
  -a 0 \
  input \
  &> output.log
```

Submit:

```bash
sbatch run_petar.slurm
```

Monitor:

```bash
squeue -u $USER
less output.log
```

Adjust the `module load` lines to match your server.

---

## 10. Process PeTar output

After the run finishes, gather the snapshots:

```bash
petar.data.gether data
```

Then run the standard post-processing:

```bash
petar.data.process -G 0.00449830997959438 data.snap.lst
```

Here:

```text
-G 0.00449830997959438
```

is the gravitational constant in PeTar's astronomical unit system:

```text
pc^3 / (Msun Myr^2)
```

This post-processing can detect binaries and compute cluster structural quantities such as Lagrangian and core radii.

---

## 11. Batch workflow for all suggested cases

Assume all IC text files are stored in:

```bash
/path/to/ICs
```

Create `prepare_all_cases.sh`:

```bash
#!/bin/bash
set -e

ICDIR=/path/to/ICs
RUNDIR=$PWD/runs
mkdir -p $RUNDIR

cases=(
  "core_q0p1_r1 ic_core_M1e4_M2e3_r1.txt"
  "core_q1_r1 ic_core_M1e4_M2e4_r1.txt"
  "core_q0p1_r2 ic_core_M1e4_M2e3_r2.txt"
  "cusp_q0p1_r1 ic_cusp_M1e4_M2e3_r1.txt"
  "core_BHdom_r1 ic_core_M1e5_M2e3_r1.txt"
)

for item in "${cases[@]}"; do
  set -- $item
  label=$1
  icfile=$2

  mkdir -p $RUNDIR/$label
  cd $RUNDIR/$label

  cp $ICDIR/$icfile ./ic.txt
  petar.init -v kms2pcmyr -f input ic.txt

  cat > run.sh <<'EOS'
#!/bin/bash
export OMP_STACKSIZE=128M
export OMP_NUM_THREADS=8
ulimit -s unlimited

petar -u 1 \
  -t 20.0 \
  -o 0.1 \
  -a 0 \
  input \
  &> output.log
EOS

  chmod +x run.sh
  cd - >/dev/null

done
```

Run:

```bash
chmod +x prepare_all_cases.sh
./prepare_all_cases.sh
```

Then execute one case:

```bash
cd runs/core_q0p1_r1
./run.sh
```

---

## 12. Recommended diagnostics for this BH-pair problem

For each snapshot, track the two BHs. Since the BHs were appended as the last two rows in the original IC, their PeTar particle IDs should correspond to the last two particle IDs assigned by `petar.init`.

Track:

```text
r_BH2(t)        distance of secondary BH from cluster center
r_12(t)         BH-BH separation
E_12(t)         two-body BH-BH orbital energy
a_12(t)         BH-BH semi-major axis if bound
e_12(t)         BH-BH eccentricity if bound
rho(r,t)        stellar density profile
Mstar(<r,t)     enclosed stellar mass profile
```

A practical binary-formation condition is:

```text
E_12 < 0
```

and the pair remains bound for several local dynamical times.

For the cored models, the key question is whether `r_BH2(t)` stalls near the stellar core radius instead of reaching the central BH. For the cuspy control model, the secondary should sink more efficiently if the core-stalling interpretation is correct.

---

## 13. Minimal end-to-end example

```bash
# 1. Enter one run directory
mkdir -p runs/core_q0p1_r1
cd runs/core_q0p1_r1

# 2. Copy IC
cp /path/to/ic_core_M1e4_M2e3_r1.txt ./ic.txt

# 3. Convert km/s -> pc/Myr and create PeTar input snapshot
petar.init -v kms2pcmyr -f input ic.txt

# 4. Run PeTar
export OMP_STACKSIZE=128M
export OMP_NUM_THREADS=8
ulimit -s unlimited

petar -u 1 -t 20.0 -o 0.1 -a 0 input &> output.log

# 5. Post-process
petar.data.gether data
petar.data.process -G 0.00449830997959438 data.snap.lst
```

---

## 14. Common mistakes

### Mistake 1: forgetting velocity conversion

Wrong:

```bash
petar.init -f input ic.txt
```

Correct:

```bash
petar.init -v kms2pcmyr -f input ic.txt
```

McLuster-style ICs usually store velocities in km/s, while PeTar's `-u 1` expects pc/Myr.

### Mistake 2: enabling stellar evolution for artificial BH particles

For this controlled dynamical experiment, do not use:

```bash
petar.init -s bse ...
petar --stellar-evolution ...
```

unless you explicitly redesign the particle types and compact-object treatment.

### Mistake 3: interpreting binary formation as merger

A Newtonian PeTar run can show whether the two BHs form a bound binary and harden. It does not automatically imply a physical GW merger unless the relevant compact-object/GW treatment is enabled and validated for this setup.

### Mistake 4: overusing MPI for a small dense system

Too many MPI ranks can reduce efficiency if most hard work is concentrated in one compact subsystem. Start with a modest number of ranks and threads.

---

## 15. Suggested first run order

Run in this order:

```text
1. core_q0p1_r1     fiducial cored model
2. cusp_q0p1_r1     compare against cuspy control
3. core_q0p1_r2     check inspiral from larger radius
4. core_q1_r1       test stronger secondary
5. core_BHdom_r1    test central-BH-dominated limit
```

This ordering gives the cleanest physical comparison with the smallest number of runs.
