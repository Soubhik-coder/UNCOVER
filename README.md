# UNCOVER (UNsupervised Clustering-based Oracle-less Variant Extraction and Recovery) in Input-Dependent Key-based Logic Locking (IDKLL) Source Repository

We extend our gratitude for your interest in our work. This repository contains the source code, datasets, and evaluation scripts for the UNCOVER work done. The provided tools facilitate a comprehensive pipeline for analyzing and evaluating IDKLL methodologies.

## Repository Structure

The repository is organized into the following primary directories and files. Please note that the `Scripts` directory must be maintained exactly as structured below; no file renaming is permitted, as the pipeline relies on these specific module names. 

```text
.
├── README.md
├── Datasets/
│   ├── TwoInputGates/
│   └── TwoLevels/
│       ├── CWS_Bench.py
│       ├── EBQ_Corpus.py
│       ├── EKT_Corpus.py
│       ├── HFS_Corpus.py
│       ├── IPC_Corpus.py
│       └── MEMO_Corpus.py
└── Scripts/
    ├── Evaluation/
    │   ├── BuildGraphs.py
    │   ├── DatasetScan.py
    │   └── Evaluate.py
    ├── FunctionalAnalysis/
    │   ├── CoveringAttack.py
    │   └── EvaluateCovering.py
    ├── Locking/
    │   ├── BatchLock.py
    │   ├── Database.py 
    │   ├── GateLock.py
    │   ├── VerifyLock.py
    │   └── DatabaseDirectory/
    ├── StructuralFingerprinting/
    │   ├── Adaptive.py
    │   ├── Collapse.py
    │   ├── Consensus.py
    │   ├── KeyRegion.py
    │   └── StructuralHash.py
    └── RunPipeline.py
```

## Setup and Usage Instructions

### 1. Environment and Script Configuration

To ensure the pipeline functions correctly, copy the `Scripts` directory into your working environment exactly as provided. Do not alter the directory structure or rename any files within this directory.

### 2. Dataset Configuration

The `Datasets` directory contains the databases of the locked variants used in our evaluations. It is divided into two subdirectories:

* `TwoLevels`: Contains variants corresponding to 2-level Quine-McCluskey formulations.
* `TwoInputGates`: Contains variants corresponding to two-input constrained gates.

**To configure and utilize a specific dataset:**

The pipeline architecture allows for multiple batches of databases to be configured and run. The `--dbs` argument in `RunPipeline.py` directly targets specific subdirectories that you must create under the `Scripts/Locking/` directory.

To set up a database for the pipeline:

1. Create a new directory under `Scripts/Locking/` and name it after your desired database (e.g., `Scripts/Locking/MyCorpus/`).
2. Copy `BatchLock.py`, `VerifyLock.py`, and `GateLock.py` from the base `Scripts/Locking/` directory into your newly created directory.
3. Navigate to the `Datasets` directory and select your desired variant script.
4. Copy this selected variant script into your new directory (`Scripts/Locking/MyCorpus/`) and rename it strictly to `Database.py`.

This modular setup replaces the default dummy database script and links your selected dataset to the execution pipeline. It allows you to stage multiple unique databases simultaneously, which you can then process selectively or sequentially by referencing their folder names via the `--dbs` flag (e.g., `--dbs MyCorpus`).


### 3. Custom Mappings

If your research requires custom mappings rather than the provided datasets, you can configure this directly within the active database script. After setting up your `Database.py` script as outlined above, open the file and modify the `key` and `partition_type` variables to align with your custom logic framework.

## Running the Pipeline (`RunPipeline.py`)

The primary orchestrator for the entire workflow is `RunPipeline.py`. It provides extensive parameter control over verification, generation, hashing, and evaluation. 

**Note on Individual Scripts:** All scripts within this repository contain thorough inline documentation comments. You can execute `python <script_name>.py -h` or `--help` on any individual file to access detailed information regarding its specific functionality and accepted arguments.

### Pipeline Arguments

Below is a detailed description of the arguments accepted by the master orchestrator, `RunPipeline.py`:

#### Global Parameters
* `--dbs`: **(Required)** List of target databases (e.g., `--dbs CWS_Bench`).
* `--workers`: Number of parallel workers to utilize. *(Default: 12)*
<<<<<<< HEAD
* `--circuit`: Target a specific circuit (e.g., 'c432'). *(Default: None - Processes all)* [THIS IS A WORK IN PROGRESS FEATURE]
=======
>>>>>>> adf8a7a (Updated instructions and folder structure)
* `--limit`: Limit the maximum number of graphs processed. *(Default: None)*
* `--force`: Force overwrite of existing caches and output files.
* `--skip-existing`: Skip processing for existing files or variants to save time.
* `--no-progress`: Disable all underlying TQDM progress bars in the terminal.

#### Path Parameters
* `--root-dir`: Root directory for the pipeline execution. *(Default: ".")*
* `--original-netlists`: Path to the directory containing original netlists. *(Default: "Original_ISCAS85/dot_json")*
* `--out-base`: **(Required)** Base output directory for all generated files and logs.

#### BatchLock: Generation Parameters
* `--min-pct`: Minimum locking percentage boundary. *(Default: 5.0)*
* `--max-pct`: Maximum locking percentage boundary. *(Default: 25.0)*
* `--step-pct`: Step size for iterating between min and max locking percentages. *(Default: 5.0)*
* `--metaseed`: Specific metaseed for dataset generation reproducibility.
* `--seed`: Base seed for overall deterministic reproducibility.
* `--variant-policy`: Distribution policy for generating locked variants. *(Default: "uniform")*
* `--key-port`: The name assigned to the key port injected into the netlist. *(Default: "key_input")*
* `--no-nand-nor`: Exclude NAND and NOR gates from being locking targets.

#### BatchLock: Verification Parameters
* `--verify`: Enables formal verification for the locked netlists.
* `--verifier`: Explicit path to the `VerifyLock.py` script. *(Default: None)*
* `--verify-samples`: Number of samples, forwarded to `VerifyLock.py` as `--samples`. *(Default: 2000)*
* `--verify-key-samples`: Number of key samples, forwarded to `VerifyLock.py` as `--key-samples`. *(Default: 1024)*
* `--verify-no-idkll`: Skip the IDKLL global-key check inside the verification script.
* `--verify-timeout`: Maximum per-circuit verification timeout in seconds. *(Default: None)*
* `--keep-failed`: Keep locked files stored even if they fail formal verification.

#### Structural Fingerprinting Parameters
* `--max-depth`: Hard cap on adaptive depth BFS (Breadth-First Search). *(Default: 8)*
* `--max-region-size`: Hard cap on subgraph region sizes for KeyRegion and Collapse. *(Default: 64)*
* `--wl-iter`: Number of Weisfeiler-Lehman (WL) graph hashing iterations. *(Default: 3)*
* `--digest-size`: WL digest string size limit in bytes. *(Default: 16)*

#### Consensus Clustering Parameters (Mutually Exclusive)
* `--k-values`: Pass an explicit list of K values (e.g., `--k-values 2 4 6`).
* `--k-range`: Define a range for K values using MIN MAX STEP (e.g., `--k-range 2 10 2`).
* *(If neither K argument is provided, the script defaults to K-values: 2, 4, 6, 8, 10)*
* `--max-cluster-size-consensus`: Ignore voting on clusters larger than this threshold to prevent combinatorial explosion. *(Default: 500)*

#### Unsupervised Evaluation & Covering Parameters
* `--eval-equivalences`: Subset of evaluation equivalences to run. Options: `strict`, `gate_type`, `partition`. *(Default: runs all three)*
* `--min-cluster-size-eval`: Minimum cluster size required for binary classification. *(Default: 2)*
* `--covering-policies`: Subset of covering match policies to execute. Options: `table`, `first`, `minimal-partitions`. *(Default: runs all three)*
* `--max-cluster-size-attack`: Cap on the Breadth-First Search extraction for functional attacks. *(Default: 256)*
* `--covering-limit`: Maximum number of candidate subgraphs to evaluate during the covering phase. *(Default: None)*
