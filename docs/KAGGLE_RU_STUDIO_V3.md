# Two Worlds Studio — Kaggle GPU production v3 (русские видеоистории)

**Версия 2026-10-30.** Задача: превратить существующие шесть русских Work-историй №13–18 в готовые медиа, максимально экономя **бесплатную** квоту Kaggle. Никаких платных генераторов по умолчанию; 48 исходных сцен остаются приватными. Сериал `TWO WORLDS — THE FRACTURED REALITY` — другой проект, не смешивать.

## Что сделано, а что только подготовлено

Реализованы **приватный Kaggle worker**, безопасная подготовка kernel, пакетная обработка двух сцен за запуск, реальная поддержка FLUX.2 Klein 4B и Qwen3-TTS CustomVoice 0.6B на **предоставленных** весах, SHA256 исходных сценариев, CPU-preflight, сохранение WAV/PNG и проекта после каждого этапа, recovery ZIP для повторной сессии, отдельная сборка 1080×1920 на CPU, CLI-verification и manual GitHub Actions. Запуск локального GPU/настоящей Kaggle-сессии **не подтверждён**; без моделей и GPU генерации не было. Тестовые синтетические изображения/голоса используются только unit-тестами и не смешиваются с настоящей партией.

## Реальная квота и аппаратура

- Kaggle `NvidiaTeslaT4` в CLI — **две отдельные T4 по 16 GiB**. Это НЕ одна 32 GiB видеокарта. Исполнитель использует **одну выбранную GPU**, позволяет переназначить `--gpu-id 0/1`; не обещает настоящую параллельную генерацию одной модели на двух T4.
- Квота бесплатных GPU индивидуальна/переменная; проверить в Kaggle → Your Work → Usage. Бесконечного бесплатного GPU нет.
- Для FLUX.2 Klein 4B через Diffusers используется CUDA float16, CPU offload, 4 шага, 576×1024 по умолчанию и загрузка модели **один раз на несколько сцен**. Полученные исходники меньше 1080p — экспорт в 1080×1920 НЕ означает нативную детализацию Full HD.
- Qwen3-TTS 0.6B рендерит русские WAV и **таймкоды реально синтезированных фраз** (не фонем), отдельным GPU-проходом. Голоса необходимо прослушать. Эмоции задаются в соответствии с жанром сцены.
- При нехватке ресурсов: 1 сцена за запуск; не включать видеодиффузию Wan2.2-5B на T4 автоматически — официальная конфигурация рассчитана на 24+ GB *на одной GPU*.

## Самый простой запуск с телефона

**Однократно:**

1. Скачать приватный архив `Two_Worlds_Studio_RU_Batch03_Ready_For_Production.zip` (из этого диалога). Создать на Kaggle **Private Dataset** с названием `two-worlds-private-queue`, добавить архив. Не публиковать истории и квитанции пользователя в публичные Kaggle Datasets/GitHub.
2. Убедиться, что в GitHub Secrets Actions задан `KAGGLE_API_TOKEN` и в Repository Variables — `KAGGLE_USERNAME`. НЕ пересылать токен сюда. Для пакета `GPU` предварительно добавить вторую приватную Kaggle Dataset `two-worlds-models`, внутри которой должны быть настоящие файлы модели в `image/` и/или `voice/` (полный snapshot с `model_index.json` / `config.json`, не один файл). Альтернатива: **явная** установка `bootstrap_weights=true` с интернетом, но этот вариант тратит время на повторную скачку больших весов.
3. После review и слияния цепочки PR #21 → #22 → текущий PR v3 открыть GitHub → Actions → **TWO WORLDS — Private Kaggle RU Shorts GPU (manual)**. Сначала выбрать `phase=preflight`, `story=13`, и собственный `username/two-worlds-private-queue`. CPU-preflight не тратит GPU.
4. Если preflight подтверждает восемь исходных сцен и правильный SHA, начать `phase=image`, `first_scene=1`, `max_scenes=1` (первый GPU-тест) или 2 после успеха. Для генерации Qwen отдельно `phase=voice`. Выберите нужный `gpu_id`, а для музыки и монтажа — фазы `music`, `render` (без GPU).
5. После завершения открыть **PRIVATE** Kaggle Kernel → Output → скачать `TwoWorlds_PRIVATE_13_image.zip` или аналог. Для продолжения загрузить предыдущий ZIP в новую **приватную** Kaggle Dataset и указать её в `recovery_dataset`, плюс полный `recovery_path=/kaggle/input/.../<file>.zip`. Восстановление сверит ID и SHA оригинального текста и не заменит другие сцены.

**Важно:** один скрипт без настоящего Kaggle Dataset / разрешения модели не сможет сгенерировать фотореализм. Предварительная подготовка пакета **не отправляет** задание на Kaggle. Github Actions workflow_dispatch становится доступным на ветке `main` **после review/merge**; создавать мердж без проверки нельзя.

## Запуск из консоли (не требуется при использовании Actions)

```bash
python -m studio.kaggle_shorts prepare \
  --username YOUR_KAGGLE_USERNAME \
  --private-dataset YOUR_KAGGLE_USERNAME/two-worlds-private-queue \
  --model-dataset YOUR_KAGGLE_USERNAME/two-worlds-models \
  --story 13 --phase image --start 1 --count 2 \
  --out /tmp/two_worlds_private_kernel \
  --install-deps
# Следующая строка РЕАЛЬНО расходует квоту — запускать только после подготовки приватных датасетов:
kaggle kernels push -p /tmp/two_worlds_private_kernel
kaggle kernels status YOUR_KAGGLE_USERNAME/two-worlds-russian-short-gpu
kaggle kernels output YOUR_KAGGLE_USERNAME/two-worlds-russian-short-gpu -p /private/output
python -m studio.kaggle_shorts verify --out /private/output --story 13
```

Если модели ещё не загружены на Kaggle, можно явно использовать `--bootstrap --install-deps --model-path /kaggle/working/private_models`, **не** передавая `--model-dataset`. Требует интернета; временные загрузки не переживают удаление окружения, поэтому для повторного использования лучше закрепить веса в PRIVATE model dataset.

## Как беречь GPU-квоту

1. **Preflight CPU первым** — ловит ошибки входного ZIP и хеша до выделения T4.
2. Короткий первый тест (1 сцена), затем 2–4 сцены, не загружать заново 4B модель на каждом кадре в одном проходе.
3. Фото FLUX и голос Qwen запускать в разных проходах. T4×2 не объединяет память автоматически; выбрано безопасное последовательное исполнение.
4. Использовать приватное хранилище весов, не качать FLUX каждый раз заново в каждой ephemeral-сессии Kaggle.
5. Скачивать **приватный** recovery ZIP после каждой партии. Прямого долговечного диска внутри Kaggle runtime нет.
6. CPU-подложка и FFmpeg не требуют GPU. Не держать платную T4 сессию для обычного кодирования.
7. Готовый рендер всегда `REVIEW_REQUIRED`; оценка лица, анатомии, русской дикции, музыки и лицензий — обязательна перед публикацией.

## Безопасность

- GitHub публичен: сценарии, фотографии, модельные веса и готовые WAV/MP4 туда **не коммитить**. Github Actions копирует результаты Kaggle только во временный runner для проверки; **не** загружает private zip в публичные артефакты и удаляет их после проверки.
- Kaggle kernel `is_private=true`; модели запускаются только из явно подключённых приватных датасетов / разрешённой открытой HuggingFace-модели.
- Нет платных API или автоматической публикации TikTok/Metricool.
- Прерывание между завершённым WAV и сохранением таймкодов может потребовать ручной проверки такого WAV; программа не выдумывает субтитры и не подменяет голос. Повтор с новым clean recovery-пакетом — допустим, но не перезаписывать сомнительные результаты втихую.
- Статус `COMPLETE` означает выполнение **запрошенного этапа** (например, 2 из 8 кадров), а НЕ готовую историю. Статус `PARTIAL` с созданными файлами может быть сохранён для восстановления после неудачного кадра.

## Что осталось для профессионального качества

Актуальные ограничения: **нет независимого доказательства качества FLUX/Qwen на реальной T4**, точной forced alignment по словам, устойчивого character-identity conditioning, живой диффузионной анимации людей, профессионального саундтрека или 6 проверенных MP4. Все 48 сцен всё ещё ожидают GPU-генерацию. Эти функции нельзя честно считать готовыми из одних unit-тестов.

## Лицензии и официальные источники

- FLUX.2 Klein **4B** (Apache 2.0): https://bfl.ai/models/flux-2-klein
- Qwen3-TTS CustomVoice 0.6B: https://github.com/QwenLM/Qwen3-TTS
- Kaggle kernels CLI и `NvidiaTeslaT4`: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
- Kaggle T4×2 и P100 retirement: https://www.kaggle.com/discussions/product-announcements/735239