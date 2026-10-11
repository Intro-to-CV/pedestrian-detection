# Initial FP/FN observations

The shared evaluator reports 13 false positives and 3 false negatives at the
review operating point (confidence >= 0.25, IoU >= 0.5).

The three false negatives are two pedestrians in `FudanPed00025` and one in
`PennPed00044`. These are crowded scenes with close or overlapping people,
where individual instances are harder to separate. The false-positive list is
concentrated in several crowded validation images as well; for example,
`FudanPed00016`, `FudanPed00041`, `PennPed00045` and `PennPed00093` have a
relatively high-confidence extra detection or a box that does not reach the
IoU=0.5 match requirement.

Visual inspection of `val_batch0_pred.jpg` and `val_batch1_pred.jpg` shows
that clear, medium-to-large pedestrians are usually localized well. The main
remaining failure pattern is crowded people and near-miss localization, rather
than a systematic failure to detect isolated pedestrians.
