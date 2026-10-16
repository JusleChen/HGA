# Third-party model code

The HGA forward model and source-feature integration are distinct from the
classification backbones. Backbones are used as evaluation interfaces.

- **ATCNet**: `hga/models/atcnet.py` is the project's PyTorch port used in
  the final BCIC/HGD experiments. Architecture source: H. Altaheri et al.,
  *IEEE Transactions on Industrial Informatics* (2023),
  [official EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet).
- **TCFormer**: only the official model and its required helper modules are
  included in `hga/models/tcformer/`. Imports were made package-relative;
  model equations are retained. The original MIT license is included there.
  [Official source](https://github.com/Altaheri/TCFormer).
- **MIRepNet**: the official `mlm_mask` model, channel positions and selected
  EA/interpolation/training helpers are included in `hga/models/mirepnet/`.
  The model's unused `wandb` import was removed; numerical definitions remain unchanged.
  The original MIT license is included there. No pretrained weights are
  redistributed. The final experiment used upstream commit
  `b35d113cd7b1629b1e0f35a5f2495ee15c69c291`.

Original copyright notices and third-party licenses apply to those components.
No new blanket license is assigned to the project's own code by this upload.
