# CentralControl

## Overview
Intel RealSense D455を用いたRe-IDとロボットの制御を行うプログラム

## Description



### Input and Output
Input: RealSenseのカラー画像と深度データ．
Output: ロボットへの動作指令．正面方向への移動速度と回転速度と方向．

### Algorithm etc



### Basic Information

|  |  |
----|---- 
| Module Name | CentralControl |
| Description | operate Re-ID and robot control |
| Version | 1.0.0 |
| Vendor | Kashima |
| Category | Controller |
| Comp. Type | STATIC |
| Act. Type | PERIODIC |
| Kind | DataFlowComponent |
| MAX Inst. | 1 |

### Activity definition

<table>
  <tr>
    <td rowspan="4">on_initialize</td>
    <td colspan="2">implemented</td>
    <tr>
      <td>Description</td>
      <td></td>
    </tr>
    <tr>
      <td>PreCondition</td>
      <td></td>
    </tr>
    <tr>
      <td>PostCondition</td>
      <td></td>
    </tr>
  </tr>
  <tr>
    <td>on_finalize</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>on_startup</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>on_shutdown</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td rowspan="4">on_activated</td>
    <td colspan="2">implemented</td>
    <tr>
      <td>Description</td>
      <td></td>
    </tr>
    <tr>
      <td>PreCondition</td>
      <td></td>
    </tr>
    <tr>
      <td>PostCondition</td>
      <td></td>
    </tr>
  </tr>
  <tr>
    <td rowspan="4">on_deactivated</td>
    <td colspan="2">implemented</td>
    <tr>
      <td>Description</td>
      <td></td>
    </tr>
    <tr>
      <td>PreCondition</td>
      <td></td>
    </tr>
    <tr>
      <td>PostCondition</td>
      <td></td>
    </tr>
  </tr>
  <tr>
    <td rowspan="4">on_execute</td>
    <td colspan="2">implemented</td>
    <tr>
      <td>Description</td>
      <td></td>
    </tr>
    <tr>
      <td>PreCondition</td>
      <td></td>
    </tr>
    <tr>
      <td>PostCondition</td>
      <td></td>
    </tr>
  </tr>
  <tr>
    <td>on_aborting</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>on_error</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>on_reset</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>on_state_update</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>on_rate_changed</td>
    <td colspan="2"></td>
  </tr>
</table>

### InPorts definition

#### image_data

ロボットに搭載された深度カメラで撮影したカラー画像

<table>
  <tr>
    <td>DataType</td>
    <td>RTC::CameraImage</td>
    <td></td>
  </tr>
  <tr>
    <td>IDL file</td>
    <td colspan="2">InterfaceDataTypes.idl</td>
  </tr>
  <tr>
    <td>Number of Data</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>Semantics</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>Unit</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>Occirrence frecency Period</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>Operational frecency Period</td>
    <td colspan="2"></td>
  </tr>
</table>

#### depth_data

ロボットに搭載された深度カメラで取得した深度データ

<table>
  <tr>
    <td>DataType</td>
    <td>RTC::CameraImage</td>
    <td></td>
  </tr>
  <tr>
    <td>IDL file</td>
    <td colspan="2">InterfaceDataTypes.idl</td>
  </tr>
  <tr>
    <td>Number of Data</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>Semantics</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>Unit</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>Occirrence frecency Period</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>Operational frecency Period</td>
    <td colspan="2"></td>
  </tr>
</table>


### OutPorts definition

#### motion_instruction
ロボットへの動作指令．前方向への移動速度，回転方向と速度


<table>
  <tr>
    <td>DataType</td>
    <td>RTC::TimedVelocity2D</td>
    <td></td>
  </tr>
  <tr>
    <td>IDL file</td>
    <td colspan="2">ExtendedDataTypes.idl</td>
  </tr>
  <tr>
    <td>Number of Data</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>Semantics</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>Unit</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>Occirrence frecency Period</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>Operational frecency Period</td>
    <td colspan="2"></td>
  </tr>
</table>


### Service Port definition


### Configuration definition


## Demo

## Requirement

## Setup

### Windows
ここでは，本プログラムをOpenRTP上で実行するまでの流れを説明する．ここで，`ノートPC`とはConciergeに繋いで使うノートPCのことを，`研PC`とはRe-IDやロボットの動作指令の出力を行うデスクトップPCのことを指す．
#### 前準備
1. 実行時にはOpenRTP(後述)上でノートPCと研PCを無線で接続する．接続に使うルーターのネットワークに繋ぐため，研PCが接続しているネットワークを無効にする．デスクトップ右下のインターネットアクセスをクリックしてネットワーク一覧を表示し，`ネットワークとインターネットの設定`をクリックする．
2. `アダプターのオプションを変更する`をクリックする．
3. 無効にしたいネットワークを選択し右クリックでメニューを表示し，`無効にする`をクリックする．

![画像4](https://github.com/user-attachments/assets/b718095f-95e8-4368-8eb2-65daebeb51fa)

4. ルーターの電源をONにし，起動するまで待つ．
5. ルーターが起動したら，ネットワークの一覧に`TP-Link_91E4_5G`が表示されるので，選択して接続する．(`5G`の表記が無いものは同ルーターの2G版である．)
6. `インターネットなし，セキュリティ保護あり`の状態になったら接続完了．
ノートPCの方はルーターの起動が終わり次第自動で接続されるが，接続されない場合は5.の手順で手動で接続する．

![tplink設定](https://github.com/user-attachments/assets/566f4f18-fba7-4bab-8d46-6c8f677d7fff)

#### OpenRTPの起動
1. 検索ウィンドウに`Start Naming Service`と入力し，`Start Naming Service`を起動する．起動後はコマンドプロンプトのウィンドウが表示されるが，ここで何かを操作する必要は無い．
2. 検索ウィンドウに`Open RTP`と入力し，`Eclipse SDK`を起動する．はじめにワークスペースのパスを選択する画面が表示されるので，使用するプログラム等が入っているフォルダのパスを選択して起動する．
3. 画面左側のネームサービスの欄に何も表示されていない場合は，`ネームサービスを追加`をクリックして追加する．現在操作しているPCのネームサービスを追加する場合は，アドレス入力欄に`localhost`と入力する．
ノートPCについても同様の手順を行う．
4. ノートPCと研PCを接続する．研PC側で`ネームサービスを追加`をクリックする．
5. アドレス入力欄にノートPCのIPアドレスを入力する．IPアドレスは，コマンドプロンプトで`ipconfig`と入力して表示される`IPv4アドレス`を指す．
6. ネームサービスの欄に入力したIPアドレスが表示されていたら接続ができている．





### Ubuntu

## Usage

## Running the tests

## LICENCE




## References




## Author


