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

## OpenPoseの準備(Windows) * 編集中
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


## References
> [OpenPose: Realtime Multi-Person 2D Pose Estimation using Psrt Affinity Fields](https://arxiv.org/abs/1812.08008)
> Z. Cao, G. Hidalgo Martinez, T. Simon S. Wei Y. A. Sheikh
> 2019 IEEE Transactions on Pattern Analysis and Machine Intelligence
