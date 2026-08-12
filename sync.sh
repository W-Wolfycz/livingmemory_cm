#!/usr/bin/env bash
# 同步 livingmemory_cm 到 Windows AstrBot 测试端（仅运行时必需文件）
#
# 排除：
#   .git / .venv* / __pycache__ / *.pyc / .pytest_cache / *.db / data/
#   tests/ / scripts/ / .gitignore / docs/ / requirements-test.txt / sync.sh
#
# 保留：LICENSE / NOTICE.md / README.md / CHANGELOG.md / Lucide 许可证 + 全部运行时文件

set -euo pipefail

SRC="${1:-/home/administrator/plugin/livingmemory_cm}"
DST="${2:-/mnt/c/Users/Administrator/.astrbot/data/plugins/livingmemory_cm}"

if [ ! -d "$SRC" ]; then
  echo "源目录不存在: $SRC" >&2
  exit 1
fi

for required_notice in \
  "LICENSE" \
  "NOTICE.md" \
  "README.md" \
  "CHANGELOG.md" \
  "pages/dashboard/vendor/LUCIDE_LICENSE"; do
  if [ ! -f "$SRC/$required_notice" ]; then
    echo "缺少发行所需的许可证/声明文件: $required_notice" >&2
    exit 1
  fi
done

mkdir -p "$DST"

rsync -rv \
  --exclude='.git' --exclude='.venv*' --exclude='venv/' \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.pytest_cache' --exclude='*.db' --exclude='data/' \
  --exclude='tests/' --exclude='scripts/' --exclude='docs/' \
  --exclude='requirements-test.txt' \
  --exclude='.gitignore' \
  --exclude='sync.sh' \
  "$SRC/" "$DST/"

echo
echo "同步完成: $SRC -> $DST"
