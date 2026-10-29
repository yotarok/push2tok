#!/usr/bin/env bash
# Convert the rinna/japanese-gpt2-medium SentencePiece Unigram tokenizer
# into a weighted, non-deterministic rustfst FST.
#
# Usage:
#   tools/convert_rinna.sh [output.fst]
#
# Optional environment variables:
#   REPO=/path/to/push2tok   リポジトリの場所 (既定: このスクリプトの親ディレクトリ)
#   FORMAT=binary|text       出力形式 (既定: binary)
#   MODEL=/path/spiece.model ローカルのモデルを使う場合 (既定: 固定リビジョンをダウンロード)
#   PYTHON=python3           venv の作成に使う Python
#   VENV=/path/to/venv       変換用 venv の場所 (既定: $REPO/.venv-rinna-fst)
#   PUSH_LABELS=0            出力ラベルの前方pushを無効にする (既定: 有効)
#   VERIFY=0                 変換後の動作確認を省略
#   SAMPLE=テキスト          動作確認に使う文
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT="${1:-dist/rinna-japanese-gpt2-medium.unigram.fst}"
FORMAT="${FORMAT:-binary}"
MODEL="${MODEL:-}"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$REPO/.venv-rinna-fst}"
VERIFY="${VERIFY:-1}"
PUSH_LABELS="${PUSH_LABELS:-1}"
SAMPLE="${SAMPLE:-吾輩は猫である。名前はまだ無い。}"

cd "$REPO"

if [[ ! -f tools/build_unigram_fst.py ]]; then
  echo "error: push2tok リポジトリが見つかりません (REPO=$REPO)" >&2
  exit 1
fi
if [[ "$FORMAT" != "binary" && "$FORMAT" != "text" ]]; then
  echo "error: FORMAT は binary か text を指定してください" >&2
  exit 1
fi
if [[ "$PUSH_LABELS" != "0" && "$PUSH_LABELS" != "1" ]]; then
  echo "error: PUSH_LABELS は 0 か 1 を指定してください" >&2
  exit 1
fi
if [[ "$FORMAT" == "binary" || "$PUSH_LABELS" == "1" ]]; then
  if ! command -v cargo >/dev/null 2>&1 && [[ -x "$HOME/.cargo/bin/cargo" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
  fi
  if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo が見つかりません (バイナリ形式またはラベルpushには Rust ツールチェーンが必要です)" >&2
    exit 1
  fi
fi

# 1. 変換用の Python 環境
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV"
fi
if ! "$VENV/bin/python" -c 'import sentencepiece; from sentencepiece import sentencepiece_model_pb2' >/dev/null 2>&1; then
  "$VENV/bin/python" -m pip install --quiet sentencepiece protobuf
fi

# 2. rustfst バイナリ変換用の CLI をビルド
if [[ "$FORMAT" == "binary" || "$PUSH_LABELS" == "1" ]]; then
  cargo build --release --quiet --bin unigram_fst
fi

# 3. SentencePiece Unigram -> 重み付き非決定性 FST
args=(--output "$OUTPUT" --format "$FORMAT")
if [[ "$PUSH_LABELS" == "1" ]]; then
  args+=(--push-labels)
fi
if [[ -n "$MODEL" ]]; then
  args+=(--model "$MODEL")
fi
"$VENV/bin/python" tools/build_unigram_fst.py "${args[@]}"

# 4. 動作確認: SentencePiece と FST の最良経路を比較
if [[ "$VERIFY" == "1" && "$FORMAT" == "binary" ]]; then
  "$VENV/bin/python" - "$OUTPUT" "$SAMPLE" <<'PY'
import json
import subprocess
import sys

import sentencepiece as spm

fst, text = sys.argv[1], sys.argv[2]
sp = spm.SentencePieceProcessor(model_file=fst + ".spiece.model")
result = subprocess.run(
    ["cargo", "run", "--release", "--quiet", "--bin", "unigram_fst", "--",
     "encode", fst, "-"],
    input=sp.normalize(text), text=True, capture_output=True, check=True,
)
got = json.loads(result.stdout)
want = sp.encode(text)
print("text         :", text)
print("sentencepiece:", want)
print("fst          :", got)
if got != want:
    sys.exit("verification failed: outputs differ")
print("verification OK")
PY
fi

echo
echo "生成物:"
echo "  $OUTPUT                 (rustfst VectorFst, tropical weight)"
echo "  $OUTPUT.spiece.model    (入力の正規化に使う元モデル)"
echo "  $OUTPUT.json            (ラベル規約などのメタデータ)"
