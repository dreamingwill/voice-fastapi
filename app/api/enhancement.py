from fastapi import APIRouter

router = APIRouter(prefix="/api/audio/enhancement", tags=["audio enhancement"])


@router.get("/options", summary="List audio enhancement options")
async def list_enhancement_options():
    return {
        "noiseModes": [
            {"id": "none", "label": "不降噪", "description": "直接送入识别模型。", "recommended": False},
            {"id": "classic", "label": "经典谱减", "description": "经典谱减法，轻量快速。", "recommended": False},
            {"id": "improved", "label": "改进谱减", "description": "自适应谱减，降低音乐噪声。", "recommended": True},
        ],
        "noiseStrength": {
            "min": 0.5,
            "max": 5.0,
            "default": 1.0,
            "description": "可选增强强度，1 为默认，数值越大抑制越强。",
        },
        "dereverb": {
            "label": "启用 Dereverb（WPE）",
            "description": "基于 WPE 的混响消除，可选。",
            "defaultEnabled": False,
            "parameters": {"delay": 3, "taps": 10, "iterations": 3},
        },
    }
