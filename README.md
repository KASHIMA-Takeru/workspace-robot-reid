# workspace-robot-reid
Re-IDを用いた移動ロボットの制御

## CUDA Toolkit, cuDNNの準備
### CUDA Toolkitのインストール
1. [NVIDIA公式ドライバーのダウンロードページ](https://www.nvidia.com/ja-jp/drivers/)で自分が使用しているGPUに合ったドライバーをダウンロードする．
2. ダウンロードされたexeファイルを開き，指示に従ってインストールを進める．
3. [CUDA Toolkitのダウンロードページ](https://developer.nvidia.com/cuda-toolkit-archive)で目的のバージョンを探しクリックする．
4. OSやバージョンなどを選択し，表示された`> Base Installer`の`Download`をクリックする．
5. ダウンロードされたexeファイルを開き，指示に従ってインストールを進める．

### cuDNNのインストール
1. [ダウンロードページ](https://developer.nvidia.com/rdp/cudnn-archive)でバージョンを選択する．
2. ダウンロードされたexeファイルを開き，指示に従ってインストールを進める．

### 環境変数の設定
CUDA ToolkitとcuDNNのインストールが完了したら，環境変数を設定する必要がある．ここでは，CUDA Toolkitのバージョン11.3を例に説明する．
1. デスクトップ画面左下の検索ウィンドウに`環境変数`と入力し，`環境変数を編集`を開く．
2. ユーザー環境変数の`新規`をクリックする．
3. `変数名`にCUDA_PATH，`変数値`に`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3`を入力する．このとき，末尾のv11.3はインストールされているバージョンを表しているので，異なるバージョンをインストールした場合は正しいバージョンにする．実際は`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`の下にインストールされたバージョンのフォルダが存在しているので，そのフォルダのパスをそのままコピーすればよい．
4. `OK`をクリックして環境変数を登録する．
5. 再び`新規`をクリックする．
6. 今度は`変数名`に`CUDNN_PATH`を入力し，`変数名`はCUDA_PATHの値と同じものにして`OK`をクリックし登録する．
7. ユーザーの環境変数の一覧に登録した2つの変数があることを確認し右下の`OK`をクリックする．

## Pythonの準備
1. Python3.9をインストールする．その他のバージョンでの動作確認は未実施．
2. [PyTorchドキュメント](https://pytorch.org/get-started/previous-versions)を参考にPyTorchをインストールする．下のコマンドでは，PyTorch1.11.0+cu113をインストールしている．
```console
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

`[WinError 5] アクセスが拒否されました`となった場合は，以下のように末尾に`-- user`と追記する．
```console
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 -- user
```

## OpenPoseの準備(Windows)
[OpenPoseドキュメント](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#compiling-and-running-openpose-from-source)
### Clone OpenPose
1. Windows Powershellを起動し，以下のコマンドでGithubにあるOpenPoseのリポジトリをクローンする．
```console
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose/
git submodule update --init --recursive --remote
```

### CMake Configuration
1. CMakeが未ダウンロードの場合は，[CMake](https://cmake.org/download/)へアクセスしCMakeをダウンロードする．
2. OpenPoseのフォルダーへ移動し，"build"フォルダーを作成しておく．
3. CMake-GUIを起動する．(コマンド入力またはダブルクリックで起動)
```console
#{}の中身は各々異なる
cd {OpenPoose_folder}
mkdir build/
cd build
cmake-gui..
```
4. "Where is the source code: "の欄にOpenPoseのディレクトリを選択．
5. "Where to build the binaries: "の欄に先ほど作成したbuildフォルダのパスを入れる．
この時点で`build`フォルダが存在しない場合はフォルダを作成するか聞かれるので，`Yes`を選択すればよい．

![git_cmake_gui](https://github.com/user-attachments/assets/6ec9815a-8a4a-44b0-9217-ec9b163c03df)

6. `Configure`をクリックする．
7. Visual Studioのバージョンを選択し，`Optional platform`で`x64`を選択し，`Finish`をクリックする．このとき，`Optional platform`が未選択の場合は`x64`が選択される．`Finish`押下後はConfiguringが終わるまで数分待つ．

![git_openpose_cmake_configure](https://github.com/user-attachments/assets/0a26ceca-455a-45ec-abbd-f16c04475be2)


8. Configuringが終了したらエラー箇所が赤く表示されるので，それらを修正していく．`BUILD_PYTHON`にチェックを入れて再び`Configure`をクリックする．
9. `PYTHON_EXECUTABLE`の値を正しいPython.exeのパスに置き換え，`PYBIND11_PYTHON_VERSION`に使用するPythonのバージョンを入力する．その後再び`Configure`をクリックする．
10. ウィンドウ下部に`Configuring done`と表示されたら`Generate`をクリックする．
    
![git_openpose_cmake_build_python](https://github.com/user-attachments/assets/011e83bd-e95c-42f5-a210-4d5f6e9cc2b4)
![git_openpose_cmake_python_executable](https://github.com/user-attachments/assets/2543e45d-5075-41f9-a833-d8b4173c6e57)
![git_openpose_cmake_python_version](https://github.com/user-attachments/assets/a5ffd9b5-17af-463b-8b15-fc8b00d87184)
![git_openpose_cmake_generating](https://github.com/user-attachments/assets/77192421-463c-4469-8729-01fa04e23dde)

11. ウィンドウ下部に`Generating done`と表示されたら`Open Project`をクリックしてVisual Studioを起動する．
12. Visual Studioが開いたら，上部のシステム構成を`Debug`から`Release`に変更する．

![release](https://github.com/user-attachments/assets/a6b5f134-e0c3-4fc1-aebb-5dacbb21107a)

13．`ビルド`のタブをクリックし，`ソリューションのビルド`をクリックしてビルドする．

![build_solution](https://github.com/user-attachments/assets/be9ba9dd-87a8-42c0-9eb9-f153abae6f65)

## OpenPoseの使い方
ここでは，OpenPoseをPythonプログラム内の使用方法について記述する．

### pyopenposeのインポート~OpenPoseの準備

```console
import sys
import os
import os.path as osp

#OpenPoseのフォルダのパス
dir_path = r'../../..'

#importするモジュールの対象にOpenPoseのフォルダを追加する
sys.path.append(dir_path + r'\openpose\build\python\openpose\Release');
#環境変数にOpenPosenoパスを追加
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + r'\openpose\build\x64\Release;' + \
    dir_path + r'openpose\build\bin;'

import pyopenpose as op

#Custom Params
params = dict()
params["model_folder"] = osp.join(dir_path, r'openpose\models')

#Start OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()
```

`ImportError: DLL load failed while importing pyopenpose: 指定されたモジュールが見つかりません`というエラーが出た場合は，`import pyopenpose as op`の前に次のコードを追記する．

```console
import ctypes

ctypes.cdll.LoadLibrary(osp.join(dir_path, r'openpose\build\x64\Release\openpose.dll'))
```

### OpenPoseによるキーポイントの検出
画像内の人物の関節等(=キーポイント)を検出し，それらの座標値およびキーポイントを結んだ画像を取得する．

```console
import cv2

~~
import pyopenpose as op
~~
datum = op.Datum()

#入力画像
image = cv2.imread('sample.jpg')

datum.cvInputData = image
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

#検出結果
#検出したキーポイントの座標
keypoints = datum.poseKeypoints
#キーポイントを点で表示し，隣り合うキーポイントを線で結んだ画像
keyimage = datum.cvOutputData

#結果表示
print("keypoints > ", keypoints)
cv2.imshow("keyimage", keyimage)
cv2.waitKey(0)
```

出力結果
```console
[[[1.94083908e+02 1.33996445e+02 8.92970026e-01]
  [1.99487411e+02 1.32718185e+02 8.04557920e-01]
  [1.55330917e+02 1.24704468e+02 7.86909282e-01]
  [1.36665756e+02 1.95419785e+02 7.52564192e-01]
  [1.58027374e+02 2.19427322e+02 7.97570288e-01]
  [2.50109940e+02 1.38033691e+02 7.48797059e-01]
  [2.40847504e+02 2.07466614e+02 7.59477377e-01]
  [1.95414062e+02 2.23495911e+02 6.80322230e-01]
  [1.87394424e+02 2.23479172e+02 6.79916024e-01]
  [1.62039948e+02 2.19453629e+02 6.17797673e-01]
  [1.29975479e+02 2.32809647e+02 7.69297421e-01]
  [1.04654526e+02 3.47641541e+02 6.78916574e-01]
  [2.12811172e+02 2.28794540e+02 6.52636766e-01]
  [2.26131989e+02 2.51469757e+02 7.87202954e-01]
  [2.07388489e+02 3.70326843e+02 6.70036614e-01]
  [1.84722351e+02 1.23285583e+02 9.00712550e-01]
  [2.04758484e+02 1.23351830e+02 9.01224196e-01]
  [1.67447952e+02 1.12656120e+02 6.69620574e-01]
  [2.19435959e+02 1.15303787e+02 8.10916126e-01]
  [1.92775177e+02 4.06362946e+02 6.76821470e-01]
  [2.06129028e+02 4.05042999e+02 6.89177513e-01]
  [1.99464539e+02 3.79662628e+02 3.82396191e-01]
  [5.92947693e+01 3.89044617e+02 5.88565826e-01]
  [5.92422447e+01 3.83697296e+02 5.03433168e-01]
  [1.15346725e+02 3.58289917e+02 5.52473605e-01]]

 [[4.42414581e+02 1.43320724e+02 9.22113776e-01]
  [4.73149689e+02 1.76692657e+02 8.40401471e-01]
  [4.19701324e+02 1.75366745e+02 7.00150073e-01]
  [4.21063538e+02 2.59473724e+02 7.42715657e-01]
  [4.53173859e+02 1.87424194e+02 8.11171412e-01]
  [5.23883606e+02 1.76793777e+02 7.12811708e-01]
  [5.50652527e+02 2.66220245e+02 7.75073647e-01]
  [4.95862793e+02 2.92874390e+02 8.47158134e-01]
  [4.77172913e+02 2.84873047e+02 6.23424232e-01]
  [4.49096649e+02 2.83589325e+02 6.10011220e-01]
  [4.11701447e+02 2.83538513e+02 8.67968559e-01]
  [3.78388123e+02 4.30412109e+02 7.31167555e-01]
  [5.07888000e+02 2.84915955e+02 5.82876682e-01]
  [5.69280823e+02 3.15537537e+02 8.62536669e-01]
  [5.38591858e+02 4.61138885e+02 4.69084769e-01]
  [4.41113983e+02 1.30008316e+02 8.37697923e-01]
  [4.55792786e+02 1.34072342e+02 9.12139177e-01]
  [0.00000000e+00 0.00000000e+00 0.00000000e+00]
  [4.87884674e+02 1.43420044e+02 8.42689395e-01]
  [5.33232056e+02 4.77182404e+02 1.00543074e-01]
  [5.45242188e+02 4.87835083e+02 1.56868368e-01]
  [5.29218811e+02 4.73148926e+02 2.17033103e-01]
  [3.55631317e+02 4.62470276e+02 6.05390072e-01]
  [3.48986908e+02 4.54500092e+02 5.85712731e-01]
  [3.86374756e+02 4.39779175e+02 5.56317627e-01]]

 [[3.24924988e+02 1.64721619e+02 9.06849802e-01]
  [3.40978027e+02 1.70068451e+02 7.72238612e-01]
  [2.94283905e+02 1.55380447e+02 8.00600708e-01]
  [2.70196655e+02 2.18111038e+02 6.49058640e-01]
  [2.78213806e+02 2.40845245e+02 3.53739589e-01]
  [3.85029053e+02 1.79418365e+02 7.22033679e-01]
  [3.93016418e+02 2.50122589e+02 7.41930008e-01]
  [3.57007660e+02 2.54160980e+02 8.38505805e-01]
  [3.16973145e+02 2.62166656e+02 6.94987237e-01]
  [2.92864014e+02 2.55501389e+02 6.08065069e-01]
  [2.59497681e+02 2.72815002e+02 7.99437642e-01]
  [2.59525482e+02 4.19731506e+02 4.81645226e-01]
  [3.46297119e+02 2.67527069e+02 6.37515903e-01]
  [3.23590607e+02 2.94243347e+02 8.42997372e-01]
  [2.59486359e+02 4.19747650e+02 4.35707152e-01]
  [3.15663086e+02 1.53990295e+02 8.67547393e-01]
  [3.35608856e+02 1.55399445e+02 9.02879059e-01]
  [0.00000000e+00 0.00000000e+00 0.00000000e+00]
  [3.58282562e+02 1.39376907e+02 8.12113345e-01]
  [2.10151672e+02 4.47766144e+02 3.77214491e-01]
  [2.18154709e+02 4.51758636e+02 3.98990661e-01]
  [2.71529572e+02 4.31768768e+02 3.19959432e-01]
  [2.11457275e+02 4.50447571e+02 3.68175566e-01]
  [2.10133575e+02 4.45082031e+02 3.54513526e-01]
  [2.72926086e+02 4.33089294e+02 2.97693729e-01]]]
```
出力画像
![openpose_result](https://github.com/user-attachments/assets/36070b3e-e1d3-4cae-ab2d-772f2aaebccc)

### 検出するキーポイントについて
OpenPoseでは，全身画像を入力すると1人あたり最大25か所のキーポイントが検出される．上のサンプル画像には3人写っており，`keypoints`には3人分の検出結果が1つのNumPy配列に入っている．配列の中身は，それぞれ`[キーポイントのx座標　y座標　信頼度]`を表している．座標は画像の左上が原点で，信頼度は0から1の値をとる．信頼度が高いほど検出したキーポイントの位置が正しいことを表し，県b出出来なかったキーポイントについては座標と信頼度が0になる．配列の各要素は次のように各部位と対応している．

|||
....|....

## References
> [OpenPose: Realtime Multi-Person 2D Pose Estimation using Psrt Affinity Fields](https://arxiv.org/abs/1812.08008)
> Z. Cao, G. Hidalgo Martinez, T. Simon S. Wei Y. A. Sheikh
> 2019 IEEE Transactions on Pattern Analysis and Machine Intelligence
