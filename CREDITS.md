# Credits and licenses

Al-Murshid (المرشد, code name jisr) is built on public datasets, open-source libraries and open fonts. Thank you to their authors.

## Dataset

**KArSL: Arabic Sign Language Database** (the only dataset used for training and testing).
Sidig, A. A. I., Luqman, H., Mahmoud, S., and Mohandes, M. (2021). KArSL: Arabic Sign Language Database.
*ACM Transactions on Asian and Low-Resource Language Information Processing*, 20(1), 1–19.
https://doi.org/10.1145/3423420 — project page: https://hamzah-luqman.github.io/KArSL/

The authors ask that users of the dataset cite the paper above. Video frames were obtained from the
public Kaggle copy `yousefdotpy/karsl-502`; the label sheet from `d3vl0per/karsl-502-labelss`;
per-sample counts from `tfmohamedyahia/kars502-pose-75`. The raw dataset is not included in this
repository.

```bibtex
@article{sidig2021karsl,
  title   = {KArSL: Arabic Sign Language Database},
  author  = {Sidig, Ala Addin I and Luqman, Hamzah and Mahmoud, Sabri and Mohandes, Mohamed},
  journal = {ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)},
  volume  = {20}, number = {1}, pages = {1--19}, year = {2021}, publisher = {ACM}
}
```

Datasets that were checked but could not be used (see `docs/vocab_audit.md`):
ArabicSL-Net (Zenodo 10.5281/zenodo.7771372, CC BY 4.0, private at the time of writing) and the
Arabic words sign language video dataset (Zenodo 10.5281/zenodo.8035320, CC BY 4.0, sample only).

## Libraries

| Library | Use | License |
|---|---|---|
| MediaPipe Tasks (Google), `hand_landmarker.task` and `pose_landmarker_lite.task` | Finding hands and body landmarks, in Python and in the browser | Apache 2.0 |
| ONNX Runtime Web (Microsoft) | Running the sign model in the browser | MIT |
| PyTorch | Training the model | BSD-style |
| NumPy, scikit-learn, Matplotlib, OpenCV | Data handling, metrics, figures | BSD / Apache 2.0 |
| Vite, Vitest, Playwright | Building and testing the web app | MIT / Apache 2.0 |

## Fonts (bundled with the app)

| Font | Use | License |
|---|---|---|
| Reem Kufi, by the Reem Kufi Project Authors (Khaled Hosny and contributors) | The name "المرشد" and the large sentence | SIL Open Font License 1.1 |
| Tajawal, by Boutros Fonts (via Google Fonts) | All other text | SIL Open Font License 1.1 |

Both fonts are packaged by Fontsource (`@fontsource/reem-kufi`, `@fontsource/tajawal`).
The full license texts are in `web/node_modules/@fontsource/*/LICENSE` after `npm install`.

## Design

The colors are inspired by the colors of the flag of Oman, in calm tones. The background pattern is
an original drawing of eight-pointed stars, inspired by Omani carved doors and silverwork.
The logo, an eight-pointed guide star, is an original drawing. No national emblem, coat of arms or flag is used.
