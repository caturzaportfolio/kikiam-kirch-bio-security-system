from __future__ import annotations

from bio_security.model_assets import SFACE, YUNET


def test_opencv_model_asset_metadata_is_pinned() -> None:
    assert YUNET.sha256 == "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    assert YUNET.size == 232589
    assert SFACE.sha256 == "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
    assert SFACE.size == 38696353
