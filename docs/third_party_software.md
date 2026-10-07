# Third-party software

Python dependencies are listed in the two requirements files. The publication requirements are the previously supplied redraw environment; the analysis requirements are inferred dependencies and need reconciliation with the actual experiment environment.

The reference-backbone code records these pinned revisions:

| Software | Revision recorded by the original runner |
| --- | --- |
| rtdl-revisiting-models | `e3ed46cac38568785289d8fa16b8cfa585bde27e` |
| TabM | `28e47ae301c92ec37787dde1ce923a0793f405b4` |

Backbone runners expect `rtdl_revisiting_models.py` and `tabm_reference.py` in their supplied vendor directory. cpDistiller runners expect a separate upstream checkout. Those external source trees are not copied into this package; their repositories, effective revisions, dependency environments and licence notices must be verified before upstream reruns or redistribution.
