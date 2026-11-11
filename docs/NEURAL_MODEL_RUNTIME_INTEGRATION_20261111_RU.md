# 🎬 TWO WORLDS — реальные нейросетевые видеодвижки (11.11.2026)

## Результат интеграции

Это **исполняемые интеграции**, а не повторная таблица рекомендаций:
- **Wan2.2 TI2V-5B**: готовый локальный ComfyUI API-граф Image-to-Video по официальному шаблону; исходное фото загружается, задания отправляются в очередь, готовый MP4 скачивается, проверяется, выдаётся SHA-256-квитанция.
- **LTX-2.5 DFR**: локальное исполнение через *экспортированный из установленного ComfyUI API-граф DFR*, со строго явными привязками полей (для официальных графов нет универсальных числовых node ID). Веса не включены.
- **Wan-Animate-2 Distilled**: отдельный локальный запуск оригинального `infer/wan_animate_2_demo.py` с управляющим видео, 10 шагами, CFG=1 и Euler согласно официальной конфигурации. Запуск ограничен 2 высокопамятными GPU по умолчанию.
- **Hybrid Master / `studio.cli produce`**: принимает подтверждённые нейроклипы с квитанциями, закрепляет референс (и управляющее видео для Wan-Animate-2), включает клип в кеш по хешам, требует независимый контроль движения персонажа в master.

Ни один путь **не** генерирует поддельные кадры, не дублирует замороженный последний кадр вместо полноценного актёрского движения, не скачивает тайно веса, не платит по внешним API и не публикует ролики. Репозиторий публичный: хранить реальные портреты/диалоги/веса/MP4 нужно **вне Git**.

⚠️ **Важно:** CI тестирует API-контракт на имитациях и синтетических MP4. Фактическое качество новых нейросетей станет проверяемым только после реальной генерации на оборудовании владельца.

## 1. Wan2.2 5B: практический режим для Kaggle T4 / локального ComfyUI

Установи текущий [ComfyUI](https://github.com/comfyanonymous/ComfyUI) **в частном окружении** и законно скачай официальные веса:
- `ComfyUI/models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors`
- `ComfyUI/models/vae/wan2.2_vae.safetensors`
- `ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`

Инструкция и веса: https://docs.comfy.org/tutorials/video/wan/wan2_2

**Скопировать официальные веса в приватный ComfyUI можно теперь штатным установщиком студии.** По умолчанию он показывает только список необходимых файлов; сетевое скачивание запускается строго отдельным флагом:

```bash
# Без сети, GPU и изменений файлов:
python -m studio.model_install --model wan22-5b --comfy-root /private/ComfyUI

# Скачать и установить официальные checkpoint файлы (≈18 ГБ исходных весов):
python -m studio.model_install --model wan22-5b \
  --comfy-root /private/ComfyUI --accept-model-license --execute-download

# Для LTX-2.5 DFR — отдельное законное лицензирование и большой объём диска:
python -m studio.model_install --model ltx25-dfr \
  --comfy-root /private/ComfyUI --accept-model-license --execute-download
```

Установщик обращается только к фиксированным официальным Hugging Face репозиториям, фиксирует их commit SHA, проверяет метаданные размеров и LFS SHA (если доступен), сохраняет файлы **только** в приватном `ComfyUI/models`. Для операции скачивания нужен `pip install huggingface_hub` и, в случае gated-моделей LTX, законно полученный **read**-доступ Hugging Face. **В публичный GitHub веса не загружаются**.

Официальный шаблон, на основании которого построен API-граф:
https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_wan2_2_5B_ti2v.json

Важно: оригинальный шаблон — **UI JSON**, а интерфейс ComfyUI ожидает **API JSON**. Студия содержит построенный API-граф с активированной веткой `LoadImage → Wan22ImageToVideoLatent`, поэтому для Wan2.2 5B вручную экспортировать граф **не требуется**.

Запусти ComfyUI только внутри GPU-сервера, слушая `127.0.0.1:8188`. Не открывай неаутентифицированный API в публичный Интернет; из других машин — SSH tunnel.

```bash
# Предварительная проверка — не обращается к GPU и не загружает веса:
python -m studio.comfy_neural \
  --model wan22-5b --reference /private/shot_01.png \
  --output /private/shot_01.mp4 \
  --prompt "A realistic woman turns her head and shoulders toward the camera, natural breath and cloth, static camera" \
  --accept-model-license --frames 121 --steps 50

# Реальное выполнение после запуска локального ComfyUI и установки весов:
python -m studio.comfy_neural \
  --model wan22-5b --reference /private/shot_01.png \
  --output /private/shot_01.mp4 \
  --prompt "A realistic woman turns her head and shoulders toward the camera, natural breath and cloth, static camera" \
  --accept-model-license --frames 121 --steps 50 --execute-local-gpu
```

Предустановлено: **576×1024**, 9:16, **24 FPS**, 121 кадр, 50 шагов, CFG 5, UniPC + simple, `ModelSamplingSD3.shift=8` согласно **ComfyUI-шаблону**. Это не универсальные лучшие значения; отдельные кадры могут требовать меньшего количества шагов. В случае нехватки памяти уменьшай кадры, но не считай сдублированные кадры полноценной анимацией.

Студия отправляет `/upload/image`, `/prompt`, читает `/history/{id}`, скачивает MP4 через `/view`, проверяет видео с ffprobe, сохраняет `shot_01.receipt.json`. Результат остаётся `GENERATED_REVIEW_REQUIRED`.

## 2. LTX-2.5 DFR: качество, синхронизация аудио и видео

Документация и исходники: https://github.com/Lightricks/LTX-2

Официальные требования: отдельная **LTX-2.x Community License** на веса; код — Apache-2.0. Установи актуальную версию ComfyUI с официальным DFR-workflow и компонентами LTX-2.5 на достаточно мощном GPU. Нельзя просто заменить старые файлы LTX 0.9.6 на 22B-weights: модели несовместимы.

После настройки графа:

1. Открой официальный DFR workflow и выбери `File → Export Workflow (API)`.
2. Сохрани экспорт как `/private/ltx25_api.json` (НЕ обычный workflow UI JSON).
3. Создай `/private/ltx25_bindings.json` с реальными идентификаторами нод/полей твоего графа. Все шесть полей обязательны: `reference`, `prompt`, `seed`, `width`, `height`, `frames`.
4. Убедись, что граф завершается одним `SaveVideo` (MP4, не WebM); пользователю запрещены неподтверждённые облачные/premium ноды.
5. Запусти:

```bash
python -m studio.comfy_neural \
  --model ltx25-dfr --template /private/ltx25_api.json \
  --bindings /private/ltx25_bindings.json \
  --reference /private/shot_02.png \
  --output /private/shot_02.mp4 \
  --prompt "A character turns toward the camera; realistic facial motion and cinematic lighting" \
  --frames 121 --seed 42 --accept-model-license --execute-local-gpu
```

**Пример структуры bindings, идентификаторы только для иллюстрации:**
```json
{
  "reference": ["15", "image"],
  "prompt": ["17", "text"],
  "seed": ["21", "seed"],
  "width": ["23", "width"],
  "height": ["23", "height"],
  "frames": ["23", "length"]
}
```

Не выдумывай ID при реальной настройке: бери только из API-экспорта. Студия проверяет наличие нод и полей и прекращает выполнение при несоответствии.

Два режима — **Wan** и **LTX** — могут использовать один безопасный ComfyUI-клиент и единый выпускной контракт. Встроенного универсального DFR-графа нет, поскольку его интерфейс и версии нод меняются; официальный шаблон оставлен владельцем, без сомнительного автоматического переопределения параметров.

## 3. Wan-Animate-2: управляемая анимация персонажа

Официальные код/веса и условия: https://github.com/Wan-Video/Wan-Animate-2

```bash
git clone --recursive https://github.com/Wan-Video/Wan-Animate-2.git
# Выполни оригинальные install и checkpoint инструкции.
# Проверь model_path и остальные пути в infer/wan_animate_2_distillation.yaml.

python -m studio.wan_animate2_run \
  --runtime /private/models/Wan-Animate-2 \
  --reference /private/character.png \
  --driver /private/real_actor_motion.mp4 \
  --output /private/shot_03.mp4 \
  --prompt "A young woman with natural face, black jacket, medium shot, indoor neutral lighting" \
  --width 480 --height 848 --frames 81 \
  --accept-model-license --execute-local-gpu
```

**Из исходной модели:** 10 шагов, CFG=1.0, Euler. Для 480p разработчики указывают тестирование **на 2× A800**. Студия блокирует запуск по умолчанию, если не видит минимум две карты с большим запасом памяти; **Kaggle T4 не подходит для этого проверенного сценария**.

Готовый клип, референс и исходное управляющее видео связываются хешами, создаётся `.receipt.json`. Никакой результат не выпускается автоматически.

## 4. Как подключить результат в основной видеомонтаж

**Все файлы должны находиться в каталоге рядом с монтажным `manifest.json` либо ниже его.** В журнале `clip_take.sha256` используй реальный `clip_sha256` из квитанции.

```json
{
  "schema": 1,
  "delivery_profile": "mobile",
  "visual_profile": "photoreal_cinematic",
  "shots": [{
    "clip": "shot_01.mp4",
    "duration": 5.0416666667,
    "camera": "static",
    "shot_type": "medium",
    "clip_take": {
      "engine": "neural_comfy",
      "visual_profile": "photoreal_cinematic",
      "sha256": "<SHA256_OF_ACTUAL_MP4>",
      "receipt": "shot_01.receipt.json",
      "reference": "shot_01.png"
    }
  }]
}
```

Для Wan Animate-2 используй `engine: neural_animate2` и добавь `"driver": "real_actor_motion.mp4"`. Если кадры/хеши/привязка к исходнику/длительность не совпадают, основной рендер будет остановлен.

```bash
python -m studio.cli produce \
  --manifest /private/manifest.json \
  --output /private/review.mp4 \
  --cache /private/studio_cache
```

Только после художественного просмотра, заполнения `qc_regions` с независимо измеряемыми `motion_windows`, согласования прав на видео/аудио и редакторского утверждения можно использовать `--master`. **Программе нельзя выставлять `human_approved=true` за автора.** Для фотореалистичной речи нужна отдельно проверенная синхронизация челюсти, щёк и мимики с WAV.

## Проверка

```bash
python -m unittest tests.test_comfy_neural tests.test_wan_animate2_run -v
python -m unittest discover -s tests -v
python -m studio.cli doctor
```

CI создаёт только короткие технические MP4 на CPU. Отсутствие GPU-инференса у владельца или невыданные лицензии **не** маскируются высоким статусом.

## Юридические и эксплуатационные ограничения

- Не копировать веса чужих **закрытых** моделей, ключи, авторизованные кэши или защищённые API.
- Устанавливать **официальные** модели отдельно от публичного GitHub, проверив лицензии на код и веса (они могут различаться).
- 2× T4 ≠ один GPU с 32 ГБ: память автоматически не объединяется.
- Не публиковать частные рисунки, данные персонажей, речь, сценарии и токены.
- Не выполнять никакого платного или расходующего квоту облачного задания без явного запуска владельцем.
