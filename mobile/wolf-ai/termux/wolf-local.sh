#!/data/data/com.termux/files/usr/bin/bash
# WOLF AI local Termux model manager. No paid API, token, or device root.
set -euo pipefail
umask 077
ROOT="$HOME/.wolf-ai"
DIR="$ROOT/models"
PIDFILE="$ROOT/llama.pid"
LOGFILE="$ROOT/llama.log"
SELECTED="$ROOT/model-choice"
mkdir -p "$DIR"
die() { printf 'WOLF AI: %s\n' "$*" >&2; exit 1; }
check_env() {
  command -v pkg >/dev/null || die "Нужен официальный Termux."
  case "$(uname -m)" in aarch64|x86_64) ;; *) die "Требуется 64-битный Android."; esac
}
choose() {
  case "$1" in
    small)
      MODELNAME="qwen2.5-coder-0.5b-instruct-q4_0.gguf"
      REPO="Qwen2.5-Coder-0.5B-Instruct-GGUF"
      EXPECTED="9739055e046d62a937e5b7879012209ef40ebea8a1569a96028de491f3f091d5"
      ;;
    balanced)
      MODELNAME="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
      REPO="Qwen2.5-Coder-1.5B-Instruct-GGUF"
      EXPECTED="cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046"
      ;;
    *) die "Доступно: small (429 МБ) или balanced (1.12 ГБ).";;
  esac
  MODEL="$DIR/$MODELNAME"
  URL="https://huggingface.co/Qwen/$REPO/resolve/main/$MODELNAME"
}
verified() {
  [[ -s "$MODEL" ]] || return 1
  printf '%s  %s\n' "$EXPECTED" "$MODEL" | sha256sum --check --status
}
choose_default() {
  if [[ -f "$SELECTED" ]]; then read -r CHOICE < "$SELECTED"; else CHOICE=small; fi
}
setup() {
  check_env
  choose "$CHOICE"
  pkg update -y
  pkg install -y curl coreutils llama-cpp
  command -v llama-server >/dev/null || die "llama-server не установлен."
  if verified; then
    echo "Модель найдена; SHA256 совпал."
  else
    rm -f "$MODEL"
    echo "Загрузка оригинального GGUF от Qwen: $MODELNAME"
    # Keep .part on interrupted downloads; reject any complete bad file.
    curl -fL --retry 3 --connect-timeout 25 -C - -o "$MODEL.part" "$URL" ||
      die "Нет доступа к Hugging Face. Частично загруженный файл сохранён."
    mv "$MODEL.part" "$MODEL"
    if ! verified; then
      rm -f "$MODEL"
      die "Контрольная SHA256 не совпала — модель отклонена."
    fi
  fi
  [[ "$(head -c 4 "$MODEL")" == GGUF ]] || die "Ошибка заголовка GGUF."
  printf '%s\n' "$CHOICE" > "$SELECTED"
  echo "✅ Локальная модель готова. Выполни: bash wolf-local.sh start"
}
health() {
  command -v curl >/dev/null || return 1
  curl -fsS --noproxy '*' --max-time 3 http://127.0.0.1:8080/v1/models >/dev/null 2>&1
}
start_model() {
  check_env
  choose "$CHOICE"
  command -v llama-server >/dev/null || die "Сначала bash wolf-local.sh setup."
  verified || die "Модель отсутствует либо повреждена. Выполни setup."
  if health; then echo "✅ Локальный ИИ уже работает."; return; fi
  nohup llama-server -m "$MODEL" --host 127.0.0.1 --port 8080 \
    --ctx-size 2048 --threads 4 --n-gpu-layers 0 \
    > "$LOGFILE" 2>&1 </dev/null &
  echo "$!" > "$PIDFILE"
  for n in $(seq 1 45); do
    if health; then
      printf '%s\n' "$CHOICE" > "$SELECTED"
      echo "✅ WOLF AI подключён к локальному ИИ 127.0.0.1:8080."
      return
    fi
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { tail -n 12 "$LOGFILE"; die "Сервер завершился."; }
    sleep 2
  done
  tail -n 12 "$LOGFILE"
  die "Модель не смогла запуститься."
}
stop_model() {
  if [[ ! -s "$PIDFILE" ]]; then echo "Нет сохранённого процесса."; return; fi
  read -r pid < "$PIDFILE"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmd" == *llama-server* && "$cmd" == *"$DIR/"* ]]; then
      kill "$pid" && echo "WOLF AI остановлен."
    else
      echo "PID больше не принадлежит этому серверу; ничего не остановлено."
    fi
  fi
  rm -f "$PIDFILE"
}
if [[ "$#" -gt 0 ]]; then ACTION="$1"; else ACTION=help; fi
choose_default
if [[ "$#" -gt 1 ]]; then CHOICE="$2"; fi
case "$ACTION" in
  setup) setup ;;
  start) start_model ;;
  stop) stop_model ;;
  restart) stop_model; start_model ;;
  status) if health; then echo "✅ LOCAL AI ONLINE"; else echo "○ LOCAL AI OFFLINE"; fi ;;
  *)
    echo "bash wolf-local.sh setup small     # 0.5B, 429 МБ"
    echo "bash wolf-local.sh setup balanced  # 1.5B, 1.12 ГБ"
    echo "bash wolf-local.sh start"
    echo "bash wolf-local.sh stop"
    echo "bash wolf-local.sh status"
    ;;
esac
