# Sort
自身の研究で必要なツール。GAN Inversionの出力結果が欲しい順番で得られないため、並べ替える必要がある。潜在ベクトル同士を比較して並べ替えることも可能ではあるが、GAN Inversionの性質上、使用する種類によって潜在ベクトルの値が異なるため難しい。そこで、再構成した画像同士を比較することで並べ替えることにした。

## 使い方
### 1. StyleGANとLPIPSの重みを取得
StyleGANとLPIPSの重みを取得する必要がある。StyleGANは、[genforce/idinvert-pytorch][indomain]のもの、LPIPSは、[richzhang/PerceptualSimilarity][lpips]のものを使用している。そのため、各リポジトリに移動し、READMEのリンクから重みをダウンロードする必要がある。

[indomain]:https://github.com/genforce/idinvert_pytorch
[lpips]:https://github.com/richzhang/PerceptualSimilarity

### 2. 実行
潜在ベクトルを並べ替える場合は,sort_code.pyを実行する。
引数は、上から順番に
- 並べ替えたい潜在ベクトルのパス
- お手本となる潜在ベクトルのパス
- 並べ替えた潜在ベクトルを保存するためのパス
- 並べ替えた潜在ベクトルを使用して画像を生成し、その画像を保存するためのディレクトリ
```
python sort_code.py \
--dlatents_path_4_sort ~.npy \
--example_path ~.npy \
--output_dir ~.npy \
--dir4img ~/images
```
