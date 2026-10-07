# Data access

This package contains computational results and source tables. It contains no new biological measurements.

| Resource | Public starting point | Required preparation |
| --- | --- | --- |
| BBBC047 / Rosetta | https://github.com/carpenter-singh-lab/2022_Haghighi_NatureMethods and Cell Painting Gallery accession `cpg0003-rosetta` | Obtain matched CP/L1000 inputs; reproduce the documented preprocessing and original compound split. |
| LINCS Cell Painting | Cell Painting Gallery accession `cpg0004-lincs` | Preserve compound-dose-plate identity, feature mapping, train-only preprocessing and frozen partitions. |
| sci-Plex3 | Srivatsan et al. (2020), using the input release identified in the manuscript | Reproduce the independent-repeat pseudobulk input, Vehicle-only feature selection and source/target cell-line roles. |

The exact sci-Plex3 input accession/release and checksums must be reconciled with the effective run records before an upstream rerun. Public access to the original studies does not by itself provide the processed arrays expected by every runner.

Dataset licences are independent of the eventual software licence. This candidate does not redistribute bulk assay data or external model checkpoints. Do not infer redistribution permission from a dataset being downloadable.
