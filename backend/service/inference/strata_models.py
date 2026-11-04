"""The GGUF files Strata can run: imported by the engine and by the model registry alike."""

from pathlib import Path

# The GGUF files Strata's setup installs from --gguf-dir, by the name of the first (or only)
# file: (--family, --model). It is setup's own list for the strata-sleep package the build
# carries (Strata 0.2.7: MODELS / FAMILIES in setup.py, "SUPPORTED_GGUFS"); setup refuses
# any other GGUF, Unsloth's UD-IQ3_XXS and the plain K-quants among them. The model list
# marks these (strata_supported), which is all the frontend offers for Strata; this is the
# one copy, so a Strata upgrade changes it here only.
STRATA_GGUF_MODELS: dict[str, tuple[str, str]] = {
    name.lower(): choice
    for name, choice in {
        "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf": ("qwen", "Q2_0"),
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf": ("qwen", "IQ2_XS"),
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf": ("qwen", "IQ3_XXS"),
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf": ("qwen", "IQ3_S"),
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf": ("swift", "IQ2_XS"),
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf": ("swift", "IQ3_XXS"),
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf": ("swift", "IQ3_S"),
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf": ("coder", "IQ1_M"),
        "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf": ("unsloth", "UD-IQ4_XS"),
        "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf": ("unsloth", "UD-Q4_K_XL"),
        "Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf": ("unsloth", "UD-Q5_K_XL"),
        "Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf": ("unsloth", "UD-Q6_K_XL"),
        "Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf": ("unsloth", "Q8_0"),
        "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf": ("gyro", "Gyro-S"),
        "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf": ("gyro", "Gyro-M"),
    }.items()
}


def strata_model_choice(gguf_path: str) -> tuple[str, str] | None:
    """(--family, --model) Strata's setup takes for this GGUF, or None when it cannot run it."""
    return STRATA_GGUF_MODELS.get(Path(gguf_path).name.lower())
