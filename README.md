# nemotron-diarization

日本語の会話をリアルタイムに「誰が何を言ったか」で文字起こしする Mac 用ツール。
話者識別は NVIDIA の Nemotron 3 Diarization、文字起こしは Qwen3-ASR または whisper。

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB)
![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-Metal-000000)
![License MIT](https://img.shields.io/badge/License-MIT-green)

```
[00:05.8-00:11.6] Speaker 2  はい、お願いします。まずバックエンドの進捗から共有しますね。
[00:12.7-00:18.4] Speaker 1  ありがとうございます。APIの変更点は仕様書に反映済みですか？
[00:18.8-00:24.0] Speaker 2  反映済みです。ただ認証周りのテストがまだ残っています。
```

## クイックスタート

```bash
# 1. NeMo-Speech.cpp を main からソースビルドする。配布バイナリ v0.1.0 は Nemotron 3 未対応
brew install cmake ninja sentencepiece abseil
curl -fsSL https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.sh | sh -s -- --source
nemo-speech pull nemotron-3-diarization

# 2. このリポジトリ。make setup が venv 作成、mlx-audio と mlx-whisper の導入、マイク補助プログラムのビルドを行う
git clone https://github.com/mugi-tea/nemotron-diarization.git && cd nemotron-diarization
make setup

# 3. 起動。Terminal.app や iTerm から直接実行し、Ctrl-C で終了。初回は ASR モデル約 2.5GB を取得する
./live.sh
```

録音と結果は `sessions/` に保存されます。発言が終わってから表示まで 5〜9 秒かかります。

## 使ってみる

```bash
./live.sh --prompt "リリース計画、仕様書、結合テスト"   # 固有名詞のヒント
./live.sh --asr whisper                              # 文字起こしを whisper に
./live.sh --fast                                     # 低遅延。切り替わり直後の精度は少し落ちる
./live.sh --no-save                                  # 録音しない

.venv/bin/python -m livediar --file recording.wav              # 16kHz mono WAV を処理
.venv/bin/python -m livediar --file recording.wav --asr none   # 話者区間だけ
.venv/bin/python tools/eval_session.py sessions/<日時>.wav     # Qwen3-ASR と whisper を発言ごとに比較
.venv/bin/python -m livediar --help                            # 全オプション
```

## リポジトリ構成

```
livediar/
  diarizer.py   NeMo-Speech.cpp の話者識別 C API を ctypes で呼ぶ
  turns.py      発言区間の確定、同時発話の重複除去、ASR に渡す範囲の決定
  asr/          文字起こしのバックエンド
  pipeline.py   全体の流れ。時刻順に整列して出力
  cli.py        コマンドライン
tools/          マイク取込みの補助プログラム、テスト音声の生成、モデル比較
tests/          単体テスト + 合成音声での統合テスト
```

## 開発

```bash
make test-unit   # モデル不要
make test        # 統合テスト込み。NeMo-Speech.cpp とモデルが必要
make fixtures    # 合成テスト音声を macOS の say で再生成
```

## ライセンス

MIT。利用しているモデルとライブラリは各ライセンスに従います。
