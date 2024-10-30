## Abstract
Similar case retrieval is a crucial aspect of the legal retrieval field, significantly contributing to LegalAI systems. This task aims to retrieve cases that are highly relevant to the query case, thereby enhancing the efficiency of legal practitioners. Recent methods have leveraged the rich semantic knowledge of pre-trained models, greatly improving retrieval performance. However, these methods often overlook key legal elements within the complex language structures of case texts, such as legal event information that can impact case outcomes and judgments. This oversight results in the underutilization of critical case information. To address this issue, we proposed RAEvent, a similar case retrieval contrastive framework augmented by legal event information. This framework utilizes an enhanced case event information database to provide auxiliary information for case retrieval and employs contrastive learning techniques to better extract similar features in cases. In comparison to a range of baseline approaches, the results of our experiments highlight the efficacy of our framework. Moreover, our research provides fresh perspectives and makes a valuable contribution to the ongoing studies in similar case retrieval tasks.


## System Requirements
```bash
- Python: 3.8.15
- PyTorch: 1.13.0+cu116
- Transformers: 4.39.3
```

-----------

Please create a new environment for experimentation and install the packages listed in the `requirements.txt` file using the following commands:
```bash
conda create -n SCR python=3.8.15
pip install -r requirements.txt
```

-----------
## Usage Guide


The datasets used are publicly available and quite large. Please download them separately for event extraction purposes.

Here is the structure of the reference file format:

* **LeCard** (Untreated Dataset)
* **Database** (Enhanced dataset with event information)
    * For event extraction, you can refer to open-source frameworks like OmniEvent to perform the extraction.
* **Output** (Folder for saving models)
* **Pretrain_model** (Folder for storing pre-trained models)
* **Src** (Source code)
    * `main.py` (Main function)
    * `model/model/` (Model files)
    * `result` (Folder for saving results)
    * `tools` (Training and validation functions)
    * `utils` (Utility functions)

## Doubts

Should you have any questions or require assistance, please feel free to contact us at `Trevo1zzZ.outlook.com`.
-----------

## Citation
```bash
@INPROCEEDINGS{Changyong-etal-2024, 
  author={Fan, Changyong and Lin, Nankai and Zhou, Dong and Zhou, Yongmei and Yang, Aimin},
  booktitle={The 16th Asian Conference on Machine Learning}, 
  title={A Retrieval-Augmented Contrastive Framework for Legal Case Retrieval Based on Event Information}, 
  year={2024}
}
```