# RPB-v23 consecutive-change candidate

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.

The separately tagged RPB-v23 wraps unchanged RPB-v22 level/content-relation
paths and adds zero-start consecutive-change moments to the learned native
coordinates. Native32 remains20 learned+12 literal fixed spectral coordinates.
The public input/output contract remains `[B,C,H,F] -> [B,D]`, with explicit
observed masks, semantic IDs, endpoints and sampling interval. No PCA follows
the encoder. Historical defaults, v22 checkpoints and synthetic results remain
preserved.

The six `consecutive_change_*.h` portable headers, artificial CUDA fixture and
`config/consecutive_changes.conf` match integration byte-for-byte. They depend
only on reusable v22/v20/model/settings code, without a market loader. Explicit
build/check targets are `rpb-v23` and `test-rpb-v23`; run them in the managed
embedding container. The integration GPU fixture is the initial engineering
execution; copied files alone do not claim another standalone test run.

Changes require two adjacent visible values and retain original-time quarters.
Unsupported/constant moments are zero; minimum support2 and RMS floor1e-6
guard arithmetic. Counts are not projected. At C3/F9, ten moments per channel/
feature form270 values and a20-by270 zero-start projection adds5400 parameters.
The fixed spectral tail and decoder are inherited. H>=17 and the base geometry
rules still apply. Distinct typed immutable inference/training formats bind
architecture, configuration, scaler and standard AdamW/counter state.

Read the [integration module](../../../../cuwacunu_dev/code/encoders/rpb_changes/README.md)
and [prospective card](../../../../cuwacunu_dev/doc/MARKET_CONSECUTIVE_CHANGE_CARD.md)
for exact formulas, masks and the single matched market development recipe.
Fresh chronological confirmation and synthetic quality remain separate work.
