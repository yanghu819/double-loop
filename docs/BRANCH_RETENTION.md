# GitHub Branch Retention

Date: 2026-08-18

This repository intentionally keeps only two active remote branches:

- `main`: GitHub default/bootstrap branch.
- `codex/gdn3-sdm-futureseed-20260816`: consolidated FutureSeed/GDN3 research branch and the only active research line.

The consolidated research state is also pinned by annotated tag
`research/futureseed-gdn3-final-20260818`.

## Meaning Of `duplicated`

Every branch in the table below is classified as `duplicated` and removed from
the remote branch namespace. The more specific evidence class is:

- `absorbed` (109): the old branch tip is an ancestor of the consolidated branch.
- `patch-equivalent` (9): the history diverged, but all branch-only patches already exist in the consolidated branch.
- `superseded-low-quality` (5): an obsolete or rejected implementation/experiment whose useful conclusions are represented by the consolidated research record.

This file is the recovery tombstone. The recorded tip SHA can be used to
reconstruct an old branch while the Git object remains available:

```bash
git branch recovered/<name> <tip-sha>
```

## Duplicated Branches

| Branch | Deleted tip SHA | Evidence class |
|---|---|---|
| `codex/cycle-edgeoff-diagnostic-20260816` | `d5bc50631f4a8831b0d486c9d80de84582a8f2aa` | `absorbed` |
| `codex/final-baseline` | `18fba54f9645f392d24cc6ff22b564e68665b4e9` | `absorbed` |
| `codex/fla-gdn2-kda` | `ab010d74685374e0cbb399ecee7cf0a3f4c37454` | `absorbed` |
| `codex/fs2-address-diagnostic-20260814` | `d09239349df3ca9a020ac9921479fc941ec0e037` | `absorbed` |
| `codex/fs2-address-mismatch-diagnostic-20260813` | `93ee1d32ece2eb535931ac5bfbcbda05f7c72935` | `absorbed` |
| `codex/fs2-loop-secant-20260815` | `4735ed93c63f605d4ea8c68d732b2cbe059d7183` | `absorbed` |
| `codex/fs2-metric-pullback-20260814` | `9f2447e882615b58cb793c80b186741657139b9d` | `patch-equivalent` |
| `codex/fs2-momentum-phase-transport-20260816` | `8ad5b0676d232ae10fe124906a2770979588069c` | `absorbed` |
| `codex/fs2-producer-readout-mqar-20260815` | `2913488bc366ec0ac2779fb84ad707b9637f0ea3` | `absorbed` |
| `codex/fs2-readonly-dual-plane-20260815` | `b5da2ac543adfeff180b94445ed0062d12cd5b78` | `patch-equivalent` |
| `codex/fs2-receiver-read-credit-20260815` | `1c603a6c570d49d16b1a21ebcae4139efb5780be` | `absorbed` |
| `codex/fs3-address-local-update-20260807` | `e20136cb58d729856059dd29b993175691045e63` | `absorbed` |
| `codex/fs3-orthogonal-basis-quality-20260812` | `8f279310d952a90ab9fa48260dc0aac90c845ff5` | `superseded-low-quality` |
| `codex/fs3-orthogonal-basis-transport-20260807` | `923a5cad34672b95b641974af0160a654c82996f` | `absorbed` |
| `codex/fs3-producer-codec-20260807` | `b015c031136ea050d0884ad58fb50184688e95e9` | `absorbed` |
| `codex/futureseed-loop-noise` | `b1bde23575d3f8c6cc79a84e90c4403d4fe9a6f1` | `absorbed` |
| `codex/futureseed-opening-projection-20260812` | `0ca20f18a67e80def0c6c7765bd4f39b498b8155` | `absorbed` |
| `codex/futureseed-phase-credit-diagnostic-20260812` | `8befd2bf018c1b940d1134ccd9ef795055c72773` | `absorbed` |
| `codex/futureseed2-block-memory-20260729` | `b644072152515f8ea71dbfc7ae53592de21a197f` | `absorbed` |
| `codex/futureseed2-content-gate-20260729` | `4bbc39382ace17de183018352858a210c31443c3` | `absorbed` |
| `codex/futureseed2-multihop-readout-20260729` | `155088200b17758f2b9122b6bc75bff85b413464` | `absorbed` |
| `codex/futureseed2-receiver-recommit-20260813` | `1ed1c1d8d76d15d3836446ffcc719018ede70c66` | `absorbed` |
| `codex/futureseed2-selective-gate-20260728` | `e8b9d5aadba62ba35a36ffb3f9ec061704945c37` | `absorbed` |
| `codex/futureseed2-state-transport-20260728` | `d1093ccc283be50b05ff8f9a980db068f14a2e0d` | `superseded-low-quality` |
| `codex/futureseed2-surprise-regression-20260813` | `9b185b640b50ccba8d4e9a6ca6e170864975c2bb` | `absorbed` |
| `codex/futureseed2-surprise-replay-20260813` | `85472b179673193a9d26a0dfc20ecda74737206d` | `absorbed` |
| `codex/gain-budget-gdn2-20260729` | `d35b6dba945798caba9f7337152be0054a0425d5` | `absorbed` |
| `codex/gdn2-address-binding-20260730` | `52edac2e13778777505736713e4845bf58748623` | `absorbed` |
| `codex/gdn2-address-carrier-20260801` | `2a9854b914f8360ea2cc4879857298138ea76407` | `absorbed` |
| `codex/gdn2-address-operator-20260731` | `c191bdbbfa0486a6d4f7e23c3f3593d56533097a` | `absorbed` |
| `codex/gdn2-binding-a100-recovery-20260812` | `4976725820d1a062ffdbe67e4dbd5d049de414ef` | `superseded-low-quality` |
| `codex/gdn2-binding-diagnostic-20260812` | `f3c28b9c9a972f862e81502a00a7cad4b364f992` | `patch-equivalent` |
| `codex/gdn2-clean-scale-12000-20260802` | `a63a1c5c8cf9f47abc61cc507739badd2d86c705` | `absorbed` |
| `codex/gdn2-fast-slow-decay-20260730` | `e1d44ec5ffb0af0e487398da1b00ff374fcfe872` | `absorbed` |
| `codex/gdn2-futureseed-precond-20260802` | `708a8c86d6194096a174f7c3ea2bb739efed4685` | `absorbed` |
| `codex/gdn3-adaptive-signed-erase-20260807` | `c09c36851e5a17224eae9d38a40923c77fe6cba2` | `absorbed` |
| `codex/gdn3-address-payload-gauge-20260815` | `97cc8f57a5c767c21466a345a6bf51bb83adb847` | `absorbed` |
| `codex/gdn3-anchored-dual-key-20260814` | `e6a39e5e30b254371f2542bdb8ded988ccd0fe52` | `patch-equivalent` |
| `codex/gdn3-atomic-pair-20260813` | `360a5a3c45d16931753add48cf7c763d931f79bf` | `absorbed` |
| `codex/gdn3-bi-axis-value-decay-20260808` | `647ebef896826259af57a8b29a3835c4f2dc5be9` | `absorbed` |
| `codex/gdn3-binding-certificate-20260815` | `4958dcfe075cecec5b3edf1cc058c1280481a90c` | `absorbed` |
| `codex/gdn3-biorthogonal-qk-mqar-20260815` | `000bc103646a794c40161a27aa932014421d5228` | `patch-equivalent` |
| `codex/gdn3-block-gram-conditioner-mqar-20260812` | `3d4242672bcaf0d0e2d1e3ba874489d946a6f479` | `absorbed` |
| `codex/gdn3-block-rls-20260815` | `7920cf5438d7de9b35dccbeb40868399bdce1c18` | `absorbed` |
| `codex/gdn3-canonical-address-companion-20260815` | `5a74fef457fd657f4d207c492af8a7aac7286acf` | `absorbed` |
| `codex/gdn3-causal-lagged-commit-20260815` | `7e2989df23b42fbf6a71dce08608c8d1f9206234` | `absorbed` |
| `codex/gdn3-chunk-local-bi-axis-20260814` | `259709e2fa296ae058ba1b63de1ce6983f0043f6` | `patch-equivalent` |
| `codex/gdn3-clustered-delta-mqar-20260812` | `d00209d9a25fb8dd4e765a5a357054e8e77b653e` | `absorbed` |
| `codex/gdn3-coherent-delta-20260807` | `f2e921bf90d63a8f221cb14895b28128caf6b933` | `absorbed` |
| `codex/gdn3-coherent-init-20260806` | `3e167b6f779065fee50b2b19107c3c5baa7088ee` | `absorbed` |
| `codex/gdn3-coherent-key-spectrum-20260815` | `8265e5fa9ab185f4ac57a7d487abaf083feaf9ca` | `absorbed` |
| `codex/gdn3-committed-delta-memory-mqar-20260812` | `cdd33f436d2dd917d2a81c129ea6aea26052d882` | `absorbed` |
| `codex/gdn3-committed-interference-credit-20260815` | `b8565ce5989cc2aae58117dec847d57c7118a985` | `absorbed` |
| `codex/gdn3-committed-residual-20260814` | `04547d67a4996d313655b342074b80b694decde1` | `patch-equivalent` |
| `codex/gdn3-contractive-dplr-20260813` | `4c051440f1c15a342dc1a05cd56d4b641f8186ff` | `absorbed` |
| `codex/gdn3-coupled-address-rows-20260807` | `c9922491c5b95ee4f3e46d0359ee78112a78469c` | `absorbed` |
| `codex/gdn3-cycle-consistency-memory-20260816` | `1df7bbb27b28f6c1720c78992292d9589c16f1b6` | `absorbed` |
| `codex/gdn3-decoupled-key-mqar-20260814` | `3b37e04f727849f66348140692e77f78f11c4f00` | `absorbed` |
| `codex/gdn3-dual-address-ceiling-20260815` | `198a8989dadf3b073d97ed80b710802bae57a3d0` | `absorbed` |
| `codex/gdn3-dual-hash-binding-mqar-20260812` | `0143504e470f6f52f6c1df0899d48d5feef9f679` | `absorbed` |
| `codex/gdn3-dynamic-frame-mqar-20260815` | `7661b90e534f9ac68a5c0688aeb0db7da3d38fc6` | `absorbed` |
| `codex/gdn3-edit-component-diagnostic-20260815` | `6702ed32b2ff10e01e9683d150e3667de91b952b` | `absorbed` |
| `codex/gdn3-gated-delta-product-20260813` | `f9d8212f00a23c2d6d05a93cf139f123496b1aa9` | `absorbed` |
| `codex/gdn3-gauge-balanced-bi-axis-20260808` | `ec0b79f5f523a3f4cad199a50064e45300db0bd9` | `absorbed` |
| `codex/gdn3-head-ownership-diagnostic-20260815` | `fe2afd0658692d2addba759444823053aec6c702` | `absorbed` |
| `codex/gdn3-interleaved-write-20260807` | `a5e8e8d7074193e40d4c3a4e1cb75c23c7465b44` | `absorbed` |
| `codex/gdn3-linear-product-state-20260815` | `1f1fbbd3621065942347f17dc3b2bf8ab9c9e478` | `absorbed` |
| `codex/gdn3-local-binding-hybrid-20260816` | `091bb622fb134050f90948049ec48203364e5280` | `patch-equivalent` |
| `codex/gdn3-log-spd-mqar-20260812` | `2af49c53eb02cc5388fbd7c2624e58db0e292d53` | `absorbed` |
| `codex/gdn3-log-spd-sudoku-20260814` | `08407336f106e03b93588c6d6bfc0529b7724dbc` | `absorbed` |
| `codex/gdn3-momentum-futureseed-20260816` | `43579c2d386e179e74107c23dec6fe9d85b2d06a` | `absorbed` |
| `codex/gdn3-momentum-log-spd-20260816` | `fcd77df31112147608d6f139ba103960a7b31f24` | `absorbed` |
| `codex/gdn3-momentum-prediction-key-20260816` | `47e4a87096da03cbc760109089aa7b85aeae0e9c` | `absorbed` |
| `codex/gdn3-native-k64-20260814` | `ad2de43406b22ab6efbbaf9605d3d55b9778eba6` | `absorbed` |
| `codex/gdn3-online-inverse-gram-20260813` | `8669fe301e95f12531105adbcee277ec4c74c26e` | `absorbed` |
| `codex/gdn3-orthogonal-chunk-state-20260807` | `8ae3105e40f44ce2526ee9bd57cfe5226a5e2499` | `absorbed` |
| `codex/gdn3-orthogonal-head-write-20260807` | `587b1068e20c4e2676ff918ac6130cc90bd559ac` | `absorbed` |
| `codex/gdn3-pair-event-encoder-20260816` | `9b7b8588dab66552f2820b3084a80349f84883f9` | `absorbed` |
| `codex/gdn3-paired-address-bank-20260807` | `8b66732cd4dce8360533b1971c67c6b5a0c4f138` | `absorbed` |
| `codex/gdn3-post-commit-momentum-refresh-20260816` | `ff06c511fbffcb8e8703824f3f2441685b58e19c` | `absorbed` |
| `codex/gdn3-predictive-residual-momentum-20260816` | `05de4da7d43eba953afc2402bb211b01ffdc9c35` | `absorbed` |
| `codex/gdn3-qk-coherence-mqar-20260815` | `9bbe0d5a9cefe07ad068dfdb017f4a5e371c7075` | `patch-equivalent` |
| `codex/gdn3-query-delta-mqar-20260815` | `c5aefd466d28607864fdfd0bb6351db2e69aa699` | `absorbed` |
| `codex/gdn3-rank2-mqar-20260812` | `18ad10fee1efa531fb970dff92efc7d985cfe07f` | `absorbed` |
| `codex/gdn3-rapid-falsifiers-20260728` | `345d99d7e10142f094c15d6e4c37640a53cd95bd` | `superseded-low-quality` |
| `codex/gdn3-raven-address-composer-20260814` | `10fd45ae32f6805e24e7058d2d56922fa52fe58f` | `absorbed` |
| `codex/gdn3-raven-routed-update-20260810` | `9b412e6416dc12fa9149b0b44a39741fe9e5a719` | `absorbed` |
| `codex/gdn3-raven-write-control-20260810` | `10cf31bf6cbcab7cda08ca3a36fad0f637308717` | `absorbed` |
| `codex/gdn3-raven-write-control-s16-20260810` | `6d2a43a2df84e26b8a4f870cd31ea19cd3c596fc` | `absorbed` |
| `codex/gdn3-redundant-address-20260815` | `943007d75ea5b8a037c2276237c8b7f4d56e8511` | `absorbed` |
| `codex/gdn3-residual-state-expert-20260807` | `15c9b530a99e25b397a6dc6798246b4a8151d485` | `absorbed` |
| `codex/gdn3-shared-eligibility-20260816` | `29688a1073b0a0dc2f3b6462bd363c8f18b9c7dc` | `absorbed` |
| `codex/gdn3-shared-log-spd-20260814` | `473235f051c52fc16d255e080b9b9383ae3623e1` | `absorbed` |
| `codex/gdn3-slot-state-20260814` | `44f107196dead62a71e24828dba192ae55cedbc0` | `absorbed` |
| `codex/gdn3-sparse-delta-slots-20260815` | `472875cdf09758a1f3e863a43af05fa6997053fd` | `absorbed` |
| `codex/gdn3-stable-token-address-20260815` | `9ef1c012055454e4838346c477a4d16e1375c59a` | `absorbed` |
| `codex/gdn3-state-feedback-update-20260807` | `2f1d84dd31c0a18ee8145041b69eff15850a4cc7` | `absorbed` |
| `codex/gdn3-terminal-consolidation-20260807` | `47ec9b2a34ef7c337f1e85ac1b09593863c2a6a4` | `absorbed` |
| `codex/gpu1-clean-backbone-baseline-20260724` | `8525da0d39dd55f1f3745bd0918c7e2a82d6f94f` | `absorbed` |
| `codex/gpu1-data-coverage` | `3bc5dec7677606cc3f12def6538891c199798ca5` | `absorbed` |
| `codex/gpu1-experiment-tracking` | `f255601753a8660a71b9079262b72c29329ee22d` | `absorbed` |
| `codex/gpu1-official-rwkv7-fs-20260724` | `5c7a4b3fe5f9c111927fc769780ea5e6e6ccaa78` | `superseded-low-quality` |
| `codex/loop-gradient-conflict-diagnostic-20260812` | `ff1a813fec2e02be64a2eaa5f40dd0c2be7dd829` | `absorbed` |
| `codex/momentum-key-locality-diagnostic-20260816` | `60d011b3f39d16fba8cb5b573ac33adf23e60080` | `absorbed` |
| `codex/mqar-native-repro-20260815` | `ab8b3b0e685beea32b16591076a31abe91d30211` | `absorbed` |
| `codex/p-causal-008-wikitext-mlm` | `cefd8814ba7275f3b8cdcdd2955108ebdc6f19d1` | `absorbed` |
| `codex/p-causal-009-established-mlm` | `a4c255e52ec2a1212d338787acd2b4f60010f3e1` | `absorbed` |
| `codex/p-causal-010-mqar-length-scaling` | `f35bc62905fc1729bde91b09b7dd636310946571` | `absorbed` |
| `codex/p-causal-011-official-bidir-mqar` | `1e63b11aea555e368b0a4d111ba3202a2612230a` | `absorbed` |
| `codex/p-causal-012-gdn2-length1024` | `22e6fd127746eb1c2f818bd9b576692d85dea49b` | `absorbed` |
| `codex/p-causal-013-gdn2-capacity1024` | `83780346ad096bebf1ef349cae56d9f54815ca57` | `absorbed` |
| `codex/p-causal-014-gdn2-value-state` | `d1fa4927591edc8898b07ed737714149839b0654` | `absorbed` |
| `codex/p-causal-015-rope-attention` | `6c9aa11c6bfc6428388deacf4e35983dd6c460a7` | `absorbed` |
| `codex/p-causal-016-gdn2-length-curve` | `d6747fbd89f2203a9cb7ce890a3b1f1348ddc342` | `absorbed` |
| `codex/p-causal-017-bidir-gdn2-ceiling` | `345d2d5e43e7c8ffb06ebfb1c5da8adb761d8f19` | `absorbed` |
| `codex/p-causal-018-pretrained-bert-carrier` | `52200d585e4a571605e650dab7889e33ee875602` | `absorbed` |
| `codex/p-causal-019-realtext-fs` | `ddca8972a4a7d457d8c66a917bc152162cb3e173` | `absorbed` |
| `codex/p-causal-020-depth-scale` | `7024e7b66f85996ea28e2a9c8eecc45347330f34` | `absorbed` |
| `codex/p-causal-021-data-diversity` | `1679e2f89f900854a3af06f0ec2f855ca017ddce` | `absorbed` |
| `codex/p-causal-022-joint-scale` | `a0e5ce77aa63b5af27a57d9b7afb3f8d3a13d594` | `absorbed` |
| `codex/p-causal-023-inference-frontier` | `f255601753a8660a71b9079262b72c29329ee22d` | `absorbed` |
| `codex/raven-futureseed-20260801` | `99b109c27820cc8fd948bffd2a5585d56a2553ee` | `absorbed` |
| `codex/sudoku-fs-gdn3-mainline` | `c2f67b69108cca049a67b8891d0d312c9b885f97` | `absorbed` |

