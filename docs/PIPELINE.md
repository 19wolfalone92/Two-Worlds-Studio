# Pipeline and acceptance checklist

**MVP**: storyboard JSON → validated scene list → procedural or licensed backgrounds → animated camera/mirror glow/particles → English captions → ambient synthetic audio → MP4.

The renderer never calls a paid API or uploads media. Input artwork is optional; fallback visuals are placeholders, not production illustrations.

## Art asset rules
- Place created or properly licensed images in `assets/`, never embed third-party watermarked content without permission.
- Set `background` per scene to a path under `assets/` (PNG/JPEG).
- Preserve character identity: script must state body and occupant separately.
- Recommended production illustrations: 1080×1920 or layered transparent PNG.

## Quality gate before any public post
- [ ] All character bodies/occupants correct for every scene.
- [ ] English text is legible on 9:16 mobile; avoid bottom UI zone.
- [ ] Real animated movement / deliberate pacing, not mislabeled as generative acting.
- [ ] No unlicensed music, logos, stock, character likenesses or dialogue.
- [ ] AI-generated video declaration if required by platform.
- [ ] Audio verified by listening, artwork checked visually.
- [ ] Manual owner approval, then scheduling in Metricool.

## Safety
Do not put TikTok/Metricool/API keys in the repository or GitHub workflow logs. Public publishing must be disabled by default. Draft preview can be generated on demand.
