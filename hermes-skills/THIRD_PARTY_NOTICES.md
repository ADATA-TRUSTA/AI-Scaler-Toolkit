# Third-party notices

## Included in this pack

### ai-short-drama-screenwriter

`skills/dramagen/references/screenwriting.md` and the prompt rules in
`skills/dramagen/story.py` are excerpted from
https://github.com/POUND0423/AI-drama-pound, used under the MIT License:

```
MIT License

Copyright (c) 2026 POUND0423

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Not included (downloaded by the user)

No model weights and no third-party ComfyUI nodes other than the pack's own
`comfyui-plugins/` are shipped. Some of the models the skills can use are not
licensed for commercial use. In particular:

- **Qwen-Image-2.1** (`imggen qwen21`, `qwen21_edit`, `qwen21_move`) and the
  **Viggle Turbo** LoRA and `viggle_turbo.py` node are under the
  **Qwen RESEARCH LICENSE AGREEMENT: non-commercial use only**.
  "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c)
  2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved."
  A commercial licence is requested from model-business@notice.qwencloud.com.

`MODELS.md` lists every model, where it comes from and its licence notes.
Check each source's licence before any commercial use.

## Installed at setup (not shipped)

`install.py` builds a separate Python environment for the `motion` skill and
installs these from PyPI; none of them is copied into this pack:

| Package | Licence | Used for |
|---|---|---|
| rapidocr-onnxruntime (with its PaddleOCR-derived ONNX models) | Apache-2.0 | finding text on screenshots by content |
| onnxruntime | MIT | running the OCR models |
| opencv-python | Apache-2.0 (OpenCV); bundled FFmpeg is LGPL | image operations behind the OCR |
| zxing-cpp | Apache-2.0 | reading and writing QR / DataMatrix codes |
| numpy, Pillow | BSD-3-Clause, MIT-CMU (HPND) | image arrays and drawing |
| rembg (optional, only for `compose.matte`) | MIT; its u2net model is Apache-2.0 | background removal |

The showcase media in `docs/showcase/` were generated with the pack itself;
every person in them is AI-generated, and the admin screen is synthetic.
