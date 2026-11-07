#!/data/data/com.termux/files/usr/bin/bash
# WOLF AI local model bootstrap for the user's own Android device.
# Does not access paid APIs, GitHub tokens or device system privileges.
set -euo pipefail
umask 077
MODEL_DIR="$HOME/.wolf-ai/models"
MODEL="$MODEL_DIR/qwen2.5-coder-0.5b-instruct-q4_0.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-q4_0.gguf?download=true"
PIDFILE="$HOME/.wolf-ai/llama-server.pid"
LOGFILE="$HOME/.wolf-ai/llama-server.log"
mkdir -p "$MODEL_DIR"
die() { echo "ОШИБКА: $*" >&2; exit 1; }
check_arch() { case "$(uname -m)" in aarch64|arm64|x86_64) : ;; *) die "Нужен 64-битный Termux";; esac; }
setup() {
  check_arch
  command -v pkg >/dev/null || die "Запусти скрипт в Termux."
  echo "Установка открытого llama.cpp без платных API..."
  pkg install -y llama-cpp curl python
  command -v llama-server >/dev/null || die "Пакет llama-cpp не содержит llama-server."
  if [[ ! -s "$MODEL" ]]; then
    echo "Загрузка Qwen2.5-Coder-0.5B-Instruct Q4_0 (~429 МБ) от авторов модели."
    curl --fail --location --retry 3 --connect-timeout 20 --continue-at - \
         --output "$MODEL.part" "$MODEL_URL" || die "Загрузка недоступна. Повтори setup позднее."
    mv "$MODEL.part" "$MODEL"
  fi
  python - "$MODEL" <<'PY'
import pathlib, sys
p=pathlib.Path(sys.argv[1])
if p.stat().st_size < 100_000_000 or p.open('rb').read(4) != b'GGUF':
    sys.exit('Недействительный GGUF; удалите повреждённый файл.')
print('GGUF файл найден:', round(p.stat().st_size / (1024**2)), 'МиБ')
PY
  echo "Модель подготовлена: $MODEL"
  echo "Следующий шаг: bash wolf-local.sh start"
}
start() {
  check_arch
  command -v llama-server >/dev/null || die "Сначала выполните setup."
  [[ -s "$MODEL" ]] || die "Модель не найдена. Сначала выполните setup."
  if curl --silent --fail --max-time 2 http://127.0.0.1:8080/v1/models >/dev/null; then
    echo "Модель уже доступна на http://127.0.0.1:8080"
    exit 0
  fi
  nohup llama-server -m "$MODEL" --host 127.0.0.1 --port 8080 \
       --ctx-size 1536 --threads 4 --n-gpu-layers 0 \
       >"$LOGFILE" 2>&1 </dev/null &
  echo $! > "$PIDFILE"
  for n in $(seq 1 45); do
    if curl --silent --fail --max-time 2 http://127.0.0.1:8080/v1/models >/dev/null; then
      echo "Локальный ИИ запущен. Открой WOLF AI на этом телефоне."
      exit 0
    fi
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || {
      tail -n 15 "$LOGFILE"; die "llama-server завершился"
    }
    sleep 2
  done
  tail -n 15 "$LOGFILE"
  die "Модель не запустилась за время проверки. Проверь свободную память."
}
stop() {
  if [[ -f "$PIDFILE" ]]; then
    pid="$(cat "$PIDFILE")"
    case "$pid" in *[!0-9]*|"") die "Некорректный PID";; esac
    if kill -0 "$pid" 2>/dev/null; then
      command_line="$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null || true)"
      if [[ "$command_line" == *llama-server* ]]; then
        kill "$pid" || true
        echo "Локальный ИИ остановлен."
      fi
    fi
    rm -f "$PIDFILE"
  fi
}
status() {
  if curl --silent --fail --max-time 3 http://127.0.0.1:8080/v1/models >/dev/null; then
    echo "RUNNING • localhost:8080"
  else
    echo "STOPPED • выполните bash wolf-local.sh start"
  fi
}
if [[ "$#" -eq 0 ]]; then action=help; else action="$1"; fi
case "$action" in
  setup) setup ;;
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) printf '%s\n' "WOLF AI (Termux)" "bash wolf-local.sh setup" "bash wolf-local.sh start" "bash wolf-local.sh status" "bash wolf-local.sh stop" ;;
esac
