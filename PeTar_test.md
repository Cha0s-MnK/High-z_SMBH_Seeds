# Running McLuster NSC + Two-BH Initial Conditions with PeTar

This guide describes how to run the NSC + two-BH systems generated with McLuster and the BH-insertion script using PeTar.

The target experiment is to test whether a secondary massive BH in a nuclear star cluster sinks to the central BH, stalls in a core, or forms a bound BH binary.

## 1. Copy dependencies & PeTar

PeTar currently depends on FDPS and SDAR. FDPS v7.0b is recommended because newer FDPS versions may cause issues in PeTar.

(Recommended directory layout) Use one parent directory for all related codes so the intended layout is:

```text
~/software/
├── FDPS-7.0b/
├── McLuster_Wang+2019/
├── PeTar/
└── SDAR/
```

Putting `FDPS`, `SDAR`, and `PeTar` in the same parent directory allows PeTar's configure script to detect the dependencies automatically.

## 2. Install PeTar

### 2.1 Configure a CPU-only pure-dynamics version

Check the configuration summary carefully. For this BH-pair experiment, start with a simple pure-dynamics build:

```bash
./configure -h
./configure \
  --prefix=$HOME/software/petar \
  --with-mpi=yes \
  --with-fdps-prefix=$HOME/software/FDPS-7.0b \
  --with-sdar-prefix=$HOME/software/SDAR
```

### 2.2 Compile and install

```bash
make -j 8
mkdir $HOME/software/petar/bin
make install
```

Add PeTar to your environment:

```bash
export PATH=$PATH:$HOME/software/petar/bin
export PYTHONPATH=$PYTHONPATH:$HOME/software/petar/include
```

For permanent setup, add the two `export` lines to `~/.bashrc`.

### 2.3 Test the installation

```bash
which petar
petar -h
petar.init -h
petar.data.gether -h
petar.data.process -h
```

## 3. Input data format & Convert IC files to PeTar format

PeTar starts from a PeTar-style snapshot file. The intermediate particle table should contain 7 columns:

```text
m  x  y  z  vx  vy  vz
```

For this project, the McLuster + BH-insertion workflow produces files like:

```text
NSCcoreM1e3M1e2r1.txt
NSCcoreM1e3M1e2r2.txt
NSCcoreM1e3M1e3r1.txt
NSCcoreM1e4M1e2r1.txt
NSCcuspM1e3M1e2r1.txt
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

Create a run directory:

```bash
mkdir -p runs/core_q0p1_r1
cd runs/core_q0p1_r1
```

Convert it to PeTar format:

```bash
cd $HOME/_ic/
petar.init -i 5 -f NSCcoreM1e3M1e2r1PeTar.txt NSCcoreM1e3M1e2r1.txt -v kms2pcmyr
petar.init -i 5 -f NSCcoreM1e3M1e2r2PeTar.txt NSCcoreM1e3M1e2r2.txt -v kms2pcmyr
petar.init -i 5 -f NSCcoreM1e3M1e3r1PeTar.txt NSCcoreM1e3M1e3r1.txt -v kms2pcmyr
petar.init -i 5 -f NSCcoreM1e4M1e2r1PeTar.txt NSCcoreM1e4M1e2r1.txt -v kms2pcmyr
petar.init -i 5 -f NSCcuspM1e3M1e2r1PeTar.txt NSCcuspM1e3M1e2r1.txt -v kms2pcmyr
```

## 4. Run a first short PeTar test

Use a short integration first to check stability:

```bash
export OMP_STACKSIZE=128M
export OMP_NUM_THREADS=8
ulimit -s unlimited
mkdir -p /lingshan/disk3/subonan/_outputs/PeTarNSCcoreM1e3M1e2r1
cd /lingshan/disk3/subonan/_outputs/PeTarNSCcoreM1e3M1e2r1
mpiexec -n 8 --bind-to none /home/subonan/software/petar/bin/petar -a 0 -b 0 -u 1 -o 0.05 -t 1.0 \
  $HOME/_ic/NSCcoreM1e3M1e2r1PeTar.txt \
  &> /lingshan/disk3/subonan/_outputs/PeTarNSCcoreM1e3M1e2r1/output.log
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

Inspect the log and check for:

```text
large energy error
large angular-momentum drift
extreme wall-clock imbalance
crashes in hard/SDAR parts
```

## 5. Run a production-scale model

For the BH-pair problem, useful first choices are:

```bash
cd /lingshan/disk3/subonan/_outputs/PeTarNSCcoreM1e3M1e2r1
nohup mpiexec -n 4 --bind-to none /home/subonan/software/petar/bin/petar -a 0 -b 0 -u 1 -o 0.05 -t 10.0 \
  $HOME/_ic/NSCcoreM1e3M1e2r1PeTar.txt > output.log 2>&1 &

export OMP_STACKSIZE=128M
export OMP_NUM_THREADS=8
ulimit -s unlimited
mkdir -p /lingshan/disk3/subonan/_outputs/PeTarNSCcoreM1e3M1e2r2
cd /lingshan/disk3/subonan/_outputs/PeTarNSCcoreM1e3M1e2r2
nohup mpiexec -n 4 --bind-to none /home/subonan/software/petar/bin/petar -a 0 -b 0 -u 1 -o 0.1 -t 20.0 \
  $HOME/_ic/NSCcoreM1e3M1e2r2PeTar.txt > output.log 2>&1 &
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

## 6. Batch workflow for all suggested cases

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

## 7. Recommended diagnostics for this BH-pair problem

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