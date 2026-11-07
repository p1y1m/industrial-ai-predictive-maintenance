# Data card — NASA C-MAPSS FD001

The bounded benchmark uses the official NASA PCoE “Turbofan Engine Degradation Simulation Data Set,” cited as A. Saxena and K. Goebel (2008). The data is high-fidelity simulated turbofan telemetry, not measurements from deployed aircraft. FD001 contains one operating condition and one HPC degradation fault mode: 100 run-to-failure training engines and 100 truncated test engines.

The archive is retrieved from NASA's official PCoE S3 URL and is not redistributed. Retrieval on 2026-11-06 produced SHA-256 `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`. NASA marks the catalog record public but does not state an explicit license for this item; users must review NASA terms before redistribution.

Quality checks enforce 26 ordered numeric columns, finite/non-null values, unique engine-cycle keys, positive integer identifiers, and contiguous cycles. Constant columns are detected from development training data only. The official test labels remain isolated until final reporting.

Limitations: simulated degradation, a single fault mode/condition, no maintenance interventions, no costs, and no causal claim. It is unsuitable for direct safety-critical decisions or aircraft certification.
