
## Experiment
### Requirements
In order to run the project please install the environment by following these commands: 
```
conda create -n TMamba python=3.10
pip install -r requirements.txt
conda activate TMamba
```

### Evaluation 
[Pretrained models] ([https://drive.google.com/drive/folders/1pVhJFwk2f3arP7zUDFAe5_PJrPSG1gc2?usp=drive_link](https://drive.google.com/drive/folders/1Zvkspmz5HiCT3M5_BgYbDlKljtgcOYcj?usp=drive_link)) <br> 


### Testing
```
python
```

### Training  
Style dataset is WikiArt collected from [WIKIART](https://www.wikiart.org/)  <br>  
content dataset is COCO2014  <br>  
```
python main.py --content_dir ./data/cnt --style_dir ./data/sty --mode train
```

## Code explanation
The full model (fig. 2(a)) can be found at [MambaST.py](https://github.com/FilippoBotti/MambaST/blob/main/models/MambaST.py). In this file you can find the whole architecture. <br>
The Mamba Encoder/Decoder (fig. 2 (b) and fig. 2 (c)) module can be found at [mamba.py](https://github.com/FilippoBotti/MambaST/blob/main/models/mamba.py) <br>
Finally, our VSSM's implementation (both with a single input and with two input merged for style transfer) can be found at [mamba_arch.py](https://github.com/FilippoBotti/MambaST/blob/main/models/mamba_arch.py). If you want you can also find VSSM with different scans direction inside [single_direction_mamba_arch.py](https://github.com/FilippoBotti/MambaST/blob/main/models/single_direction_mamba_arch.py) and [double_direction_mamba_arch.py](https://github.com/FilippoBotti/MambaST/blob/main/models/double_direction_mamba_arch.py).

### Reference
If you find our work useful in your research, please cite our paper using the following BibTeX entry ~ Thank you ^ . ^. Paper Link [pdf](https://www.arxiv.org/abs/2409.10385)<br> 


```
@InProceedings{Botti_2025_WACV,
    author    = {Botti, Filippo and Ergasti, Alex and Rossi, Leonardo and Fontanini, Tomaso and Ferrari, Claudio and Bertozzi, Massimo and Prati, Andrea},
    title     = {Mamba-ST: State Space Model for Efficient Style Transfer},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {7786-7795}
}
```

### Acknowledgments
Our code is inspired by [StyTR-2](https://github.com/diyiiyiii/StyTR-2) and [StyleID](https://github.com/jiwoogit/StyleID).
