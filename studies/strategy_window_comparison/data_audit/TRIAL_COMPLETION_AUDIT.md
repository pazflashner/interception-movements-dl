# Condition-2 trial completion audit

Trials inspected: 4,763
Participants: 28
MAT arrival present: 4,759/4,763
Arrival at/before target motion onset: 0
CSV recording ends at/before target motion onset: 0
Endpoint-cluster outliers: 13
Too early labels: 48
Too early labels with arrival at/before go: 0
Too early labels with CSV ending at/before go: 0

## Exact outcome labels

| responseText | successful | count |
|---|---:|---:|
| Success | 1 | 3006 |
| Not fixating on the dot enough!!! | -1 | 1371 |
| Too late | -3 | 337 |
| Too early | -3 | 48 |
| ???????? | -3 | 1 |

## Timing and endpoint summaries

```text
       arrival_after_go_s  late_after_window_s  csv_duration_s  go_to_end_forward  go_to_end_distance  endpoint_distance_from_subject_median
count         4759.000000          4759.000000     4763.000000        4763.000000         4763.000000                            4763.000000
mean             0.515259            -0.052368        0.854178          13.510593           13.545143                               0.750240
std              0.214561             0.217383        0.350834           0.718218            0.695347                               0.654158
min              0.167206            -0.399494        0.462500          -1.063322            0.051346                               0.005900
1%               0.317379            -0.266065        0.612500          12.407504           12.423460                               0.065016
5%               0.383824            -0.199448        0.662500          12.666457           12.690879                               0.153770
50%              0.484161            -0.079194        0.825000          13.491704           13.520071                               0.606416
95%              0.667475             0.100852        1.045833          14.492687           14.544834                               1.756634
99%              1.150642             0.607530        1.579167          15.015846           15.064268                               2.536835
max              7.150466             6.567166        9.995833          16.663723           16.955722                              17.284504
```

## Flagged cases

The machine-readable table contains 13 rows matching at least one audit flag.
No audit flag is an automatic exclusion until its event semantics are justified.