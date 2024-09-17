# workspace-robot-reid
Re-IDを用いた移動ロボットの制御

## Pythonの準備
1. Python3.9をインストールする．その他のバージョンでの動作確認は未実施．
2. [PyTorchドキュメント](https://pytorch.org/get-started/previous-versions)を参考に，PyTorch1.11.0+cu113をインストールする．
仮想環境で試したがインストールできなかったのでここは保留
```console
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
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

11. ウィンドウ下部に`Generating done`と表示されたら`Open Project`をクリックする．

