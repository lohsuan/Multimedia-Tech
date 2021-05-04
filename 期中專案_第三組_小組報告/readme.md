## How to start
Please download images in google drive we provided and put all of them in the working directory (the same directory as `predict.py`).
google drive: \
[請將此資料夾內的東西與predict.py放同一層](https://drive.google.com/drive/folders/1f2SLgnFQWOImDLP5MbW2HVFdWw5TFsu9?usp=sharing)

資料夾中除了圖片集還有使用hog 訓練的輸出模型，因檔案過大，一併放入。
### train it
```python
$ python train.py   # 若不想重新train module可直接使用google drive的模型
```

### predict and play game
```python
$ python predict.py     # input img path example: ./prediction_000/ paper_001.jpg
```

