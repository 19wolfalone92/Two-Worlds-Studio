# Private short-form production and recovery

The short-form workflow reuses the existing image, Russian voice, music,
subtitle and photographic-camera modules. It does not replace the studio.

## Boundaries

`shortform04-production.yml` is restricted to the production branch and its
own request directory. Existing integration requests and earlier publication
queues are left alone. The request-directory and branch overrides in the
relay are allowlisted, not arbitrary paths or code.

Unpublished scripts and visual prompts belong in private datasets. The
source-handshake exchange and result artifacts are encrypted; the recipient
private key stays outside this repository. Never commit tokens, cleartext
production archives, unpublished scripts, narration or frames.

## Recovering work

1. Query quota and kernel status with the read-only status phase. An optional
   progress query downloads an existing private checkpoint without asking for
   a GPU. Do not relaunch a worker just to find its progress.
2. Validate the source inventory before generating media. The existing source
   format includes the title before the scene paragraphs. All scene text and
   script hashes must match the source, including imports from checkpoints.
3. Cached model bundles are weights, not authoritative story sources. Embedded
   voice ZIPs from an old story are excluded from recovery discovery.
4. Use bounded `tasks` for voice repairs (`story`, `first_scene`, `count`).
   A partial repair must merge only the named takes and alignments; it must
   not clear the other scenes' subtitles. Two T4 devices are independent;
   their memory is not pooled.
5. Finished scene caches include source hashes, captions, camera mode and
   strength. Verify the cached file hash and duration before reusing it.
   Changing a camera move or subtitle cannot silently reuse old pixels.
6. Inspect the worker's returned status and actual assets. A completed relay
   workflow alone is not proof that a GPU worker completed every scene.

## Release checks

The renderer uses subpixel photographic camera movement and a fixed 30 FPS
master. Black-frame detection includes a single black frame; the complete
MP4 is decoded, and its duration must equal the full narration clock.
Optional opening text can be omitted when it covers a face.

Technical checks deliberately leave artistic and rights approval pending.
Review the actual story, cast continuity, props, first two seconds, subtitle
placement and ending before releasing a movie. Speech recognition is useful
for finding wording problems; it is not human listening. Check displayed
covers in the destination planner, not only the local JPG.

Keep a publication ledger with final hashes, post IDs, timezone and returned
media URLs. Reconcile the planner before retrying a create operation. A draft
is not an automatic publication, and scheduled is not published.

## Remaining autonomy work

The workflow can generate and recover media from a prepared private source.
It does not yet provide unattended original scriptwriting, editorial visual
judgment, Metricool credentials, or an automatic analytics-to-story loop.
No paid API or subscription is introduced by this change.
